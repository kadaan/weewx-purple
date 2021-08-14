# Copyright 2020 by John A Kline <john@johnkline.com>
#
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 2
# of the License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.

"""
WeeWX module that records PurpleAir air quality sensor readings.
"""

import datetime
import json
import logging
import math
import requests
import sys
import threading
import time

from dateutil import tz
from dateutil.parser import parse

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import weeutil.weeutil
import weewx
import weewx.units
import weewx.xtypes

from weewx.units import ValueTuple
from weeutil.weeutil import timestamp_to_string
from weeutil.weeutil import to_bool
from weeutil.weeutil import to_float
from weeutil.weeutil import to_int
from weewx.engine import StdService

log = logging.getLogger(__name__)

WEEWX_PURPLE_VERSION = "3.0.2"

if sys.version_info[0] < 3 or (sys.version_info[0] == 3 and sys.version_info[1] < 6):
    raise weewx.UnsupportedFeature(
        "weewx-purple requires Python 3.6 or later, found %s.%s" % (sys.version_info[0], sys.version_info[1]))

if weewx.__version__ < "4":
    raise weewx.UnsupportedFeature(
        "weewx-purple requires WeeWX 4, found %s" % weewx.__version__)

# Set up observation types not in weewx.units

weewx.units.USUnits['air_quality_index']       = 'aqi'
weewx.units.MetricUnits['air_quality_index']   = 'aqi'
weewx.units.MetricWXUnits['air_quality_index'] = 'aqi'

weewx.units.USUnits['air_quality_color']       = 'aqi_color'
weewx.units.MetricUnits['air_quality_color']   = 'aqi_color'
weewx.units.MetricWXUnits['air_quality_color'] = 'aqi_color'

weewx.units.default_unit_label_dict['aqi']  = ' AQI'
weewx.units.default_unit_label_dict['aqi_color'] = ' RGB'

weewx.units.default_unit_format_dict['aqi']  = '%d'
weewx.units.default_unit_format_dict['aqi_color'] = '%d'

weewx.units.obs_group_dict['pm2_5_aqi'] = 'air_quality_index'
weewx.units.obs_group_dict['pm2_5_aqi_color'] = 'air_quality_color'

class Source:
    def __init__(self, config_dict, name):
        # Raise KeyEror if name not in dictionary.
        source_dict = config_dict[name]
        self.enable = to_bool(source_dict.get('enable', False))
        self.sensor_id = source_dict.get('sensor_id', '')
        self.timeout  = to_int(source_dict.get('timeout', 10))

@dataclass
class Concentrations:
    timestamp       : float
    pm1_0           : float
    pm10_0          : float
    pm2_5_cf_1      : float
    pm2_5_cf_1_b    : Optional[float]
    current_temp_f  : int
    current_humidity: int
    pressure        : float

@dataclass
class Configuration:
    lock            : threading.Lock
    concentrations  : Concentrations # Controlled by lock
    archive_interval: int            # Immutable
    archive_delay   : int            # Immutable
    poll_interval   : int            # Immutable
    sources         : List[Source]   # Immutable

def datetime_from_reading(dt_str):
    dt_str = dt_str.replace('z', 'UTC')
    tzinfos = {'CST': tz.gettz("UTC")}
    return parse(dt_str, tzinfos=tzinfos)

def utc_now():
    return datetime.datetime.now(tz=tz.gettz("UTC"))

def get_concentrations(cfg: Configuration):
    for source in cfg.sources:
        if source.enable:
            record = collect_data(source.sensor_id,
                                  source.timeout,
                                  cfg.archive_interval)
            if record is not None:
                log.debug('get_concentrations: source: %s' % record)
                reading_ts = to_int(record['dateTime'])
                age_of_reading = time.time() - reading_ts
                if age_of_reading > cfg.archive_interval:
                    log.info('Reading from %s is old: %d seconds.' % (
                        source.sensor_id, age_of_reading))
                    continue
                concentrations = Concentrations(
                    timestamp        = reading_ts,
                    pm1_0            = to_float(record['pm1_0_atm']),
                    pm10_0           = to_float(record['pm10_0_atm']),
                    pm2_5_cf_1       = to_float(record['pm2_5_cf_1']),
                    pm2_5_cf_1_b     = None, # If there is a second sensor, this will be updated below.
                    current_temp_f   = to_int(record['current_temp_f']),
                    current_humidity = to_int(record['current_humidity']),
                    pressure         = to_float(record['pressure']) * 33.863886666667,
                )
                # If there is a 'b' sensor, add it in and average the readings
                log.debug('get_concentrations: concentrations BEFORE averaing in b reading: %s' % concentrations)
                if 'pm1_0_atm_b' in record:
                    concentrations.pm1_0        = (concentrations.pm1_0  + to_float(record['pm1_0_atm_b'])) / 2.0
                    concentrations.pm2_5_cf_1_b = to_float(record['pm2_5_cf_1_b'])
                    concentrations.pm10_0       = (concentrations.pm10_0 + to_float(record['pm10_0_atm_b'])) / 2.0
                log.debug('get_concentrations: concentrations: %s' % concentrations)
                return concentrations
    log.error('Could not get concentrations from any source.')
    return None

def is_type(j: Dict[str, Any], t, names: List[str]) -> bool:
    try:
        for name in names:
          x = j[name]
          if not isinstance(x, t):
              log.info('%s is not an instance of %s: %s' % (name, t, j[name]))
              return False
        return True
    except KeyError as e:
        log.info('is_type: could not find key: %s' % e)
        return False
    except Exception as e:
        log.info('is_type: exception: %s' % e)
        return False

def is_sane(j: Dict[str, Any]) -> bool:
    time_of_reading = datetime_from_reading(j['DateTime'])
    if not isinstance(time_of_reading, datetime.datetime):
        log.info('DateTime is not an instance of datetime: %s' % j['DateTime'])
        return False

    if not is_type(j, int, ['current_temp_f','current_humidity','current_dewpoint_f']):
        return False

    if not is_type(j, float, ['pressure']):
        return False

    # Sensor A
    if not is_type(j, float, ['pm1_0_cf_1','pm1_0_atm','p_0_3_um','pm2_5_cf_1',
            'pm2_5_atm','p_0_5_um','pm10_0_cf_1','pm10_0_atm']):
        return False
    if not is_type(j, int, ['pm2.5_aqi']):
        return False

    # Sensor B
    if 'pm2.5_aqi_b' in j:
        if not is_type(j, float, ['pm1_0_cf_1_b','pm1_0_atm_b','p_0_3_um_b','pm2_5_cf_1_b',
                'pm2_5_atm_b','p_0_5_um_b','pm10_0_cf_1_b','pm10_0_atm_b']):
            return False
        if not is_type(j, int, ['pm2.5_aqi_b']):
            return False

    return True

def collect_data(sensor_id, timeout, archive_interval):

    j = None
    url = 'https://www.purpleair.com/json?show=%s' % (sensor_id)

    try:
        # fetch data
        log.debug('collect_data: fetching from url: %s, timeout: %d' % (url, timeout))
        r = requests.get(url=url, timeout=timeout)
        r.raise_for_status()
        log.debug('collect_data: %s returned %r' % (sensor_id, r))
        if r:
            # convert to json
            j = r.json()
            log.debug('collect_data: json returned from %s is: %r' % (sensor_id, j))
            time_of_reading = datetime_from_reading(j['DateTime'])
            # Check for sanity
            if not is_sane(j):
                log.info('purpleair reading not sane: %s' % j)
                return None
    except Exception as e:
        log.info('collect_data: Attempt to fetch from: %s failed: %s.' % (sensor_id, e))
        j = None


    if j is None:
        return None

    # create a record
    log.debug('Successful read from %s.' % sensor_id)
    return populate_record(time_of_reading.timestamp(), j)

def populate_record(ts, j):
    record = dict()
    record['dateTime'] = ts
    record['usUnits'] = weewx.US

    # put items into record
    missed = []

    def get_and_update_missed(key):
        if key in j:
            return j[key]
        else:
            missed.append(key)
            return None

    record['current_temp_f'] = get_and_update_missed('current_temp_f')
    record['current_humidity'] = get_and_update_missed('current_humidity')
    record['current_dewpoint_f'] = get_and_update_missed('current_dewpoint_f')

    pressure = get_and_update_missed('pressure')
    if pressure is not None:
        # convert pressure from mbar to US units.
        # FIXME: is there a cleaner way to do this
        pressure, units, group = weewx.units.convertStd((pressure, 'mbar', 'group_pressure'), weewx.US)
        record['purple_pressure'] = pressure

    if missed:
        log.info("Sensor didn't report field(s): %s" % ','.join(missed))

    # for each concentration counter, grab A, B and the average of the A and B channels and push into the record
    for key in ['pm1_0_cf_1', 'pm1_0_atm', 'pm2_5_cf_1', 'pm2_5_atm', 'pm10_0_cf_1', 'pm10_0_atm']:
        record[key] = j[key]
        key_b = key + '_b'
        if key_b in j.keys():
            record[key_b] = j[key_b]
            record[key + '_avg'] = (j[key] + j[key_b]) / 2.0

    return record

class Purple(StdService):
    """Collect Purple Air air quality measurements."""

    def __init__(self, engine, config_dict):
        super(Purple, self).__init__(engine, config_dict)
        log.info("Service version is %s." % WEEWX_PURPLE_VERSION)

        self.engine = engine
        self.config_dict = config_dict.get('Purple', {})

        self.cfg = Configuration(
            lock             = threading.Lock(),
            concentrations   = None,
            archive_interval = int(config_dict['StdArchive']['archive_interval']),
            archive_delay    = to_int(config_dict['StdArchive'].get('archive_delay', 15)),
            poll_interval    = 5,
            sources          = Purple.configure_sources(self.config_dict))
        with self.cfg.lock:
            self.cfg.concentrations = get_concentrations(self.cfg)

        source_count = 0
        for source in self.cfg.sources:
            if source.enable:
                source_count += 1
                log.info(
                    'Source %d for PurpleAir readings: sensor: %s, timeout: %d' % (
                    source_count, source.sensor_id, source.timeout))
        if source_count == 0:
            log.error('No sources configured for purple extension.  Purple extension is inoperable.')
        else:
            weewx.xtypes.xtypes.append(AQI())

            # Start a thread to query proxies and make aqi available to loopdata
            dp: DevicePoller = DevicePoller(self.cfg)
            t: threading.Thread = threading.Thread(target=dp.poll_device)
            t.setName('Purple')
            t.setDaemon(True)
            t.start()

            self.bind(weewx.NEW_LOOP_PACKET, self.new_loop_packet)

    def new_loop_packet(self, event):
        log.debug('new_loop_packet(%s)' % event)
        with self.cfg.lock:
            log.debug('new_loop_packet: self.cfg.concentrations: %s' % self.cfg.concentrations)
            if self.cfg.concentrations is not None and \
                    self.cfg.concentrations.timestamp is not None and \
                    self.cfg.concentrations.timestamp + \
                    self.cfg.archive_interval >= time.time():
                log.debug('Time of reading being inserted: %s' % timestamp_to_string(self.cfg.concentrations.timestamp))
                # Insert pressure, pm1_0, pm2_5, pm10_0, aqi and aqic into loop packet.
                if self.cfg.concentrations.pressure is not None:
                    event.packet['pressure'] = self.cfg.concentrations.pressure
                    log.debug('Inserted packet[pressure]: %f into packet.' % event.packet['pressure'])
                if self.cfg.concentrations.pm1_0 is not None:
                    event.packet['pm1_0'] = self.cfg.concentrations.pm1_0
                    log.debug('Inserted packet[pm1_0]: %f into packet.' % event.packet['pm1_0'])
                if self.cfg.concentrations.pm2_5_cf_1_b is not None:
                    b_reading = self.cfg.concentrations.pm2_5_cf_1_b
                else:
                    b_reading = self.cfg.concentrations.pm2_5_cf_1 # Dup A sensor reading
                if (self.cfg.concentrations.pm2_5_cf_1 is not None
                        and b_reading is not None
                        and self.cfg.concentrations.current_humidity is not None
                        and self.cfg.concentrations.current_temp_f):
                    event.packet['pm2_5'] = AQI.compute_pm2_5_us_epa_correction(
                            self.cfg.concentrations.pm2_5_cf_1, b_reading,
                            self.cfg.concentrations.current_humidity, self.cfg.concentrations.current_temp_f)
                    log.debug('Inserted packet[pm2_5]: %f into packet.' % event.packet['pm2_5'])
                if self.cfg.concentrations.pm10_0 is not None:
                    event.packet['pm10_0'] = self.cfg.concentrations.pm10_0
                    log.debug('Inserted packet[pm10_0]: %f into packet.' % event.packet['pm10_0'])
                if 'pm2_5' in event.packet:
                    event.packet['pm2_5_aqi'] = AQI.compute_pm2_5_aqi(event.packet['pm2_5'])
                if 'pm2_5_aqi' in event.packet:
                    event.packet['pm2_5_aqi_color'] = AQI.compute_pm2_5_aqi_color(event.packet['pm2_5_aqi'])
            else:
                log.error('Found no fresh concentrations to insert.')

    def configure_sources(config_dict):
        sources = []
        # Configure Sensors
        idx = 0
        while True:
            idx += 1
            try:
                source = Source(config_dict, 'Sensor%d' % idx, False)
                sources.append(source)
            except KeyError:
                break

        return sources

class DevicePoller:
    def __init__(self, cfg: Configuration):
        self.cfg = cfg

    def poll_device(self) -> None:
        log.debug('poll_device: start')
        while True:
            try:
                log.debug('poll_device: calling get_concentrations.')
                concentrations = get_concentrations(self.cfg)
            except Exception as e:
                log.error('poll_device exception: %s' % e)
                weeutil.logger.log_traceback(log.critical, "    ****  ")
                concentrations = None
            log.debug('poll_device: concentrations: %s' % concentrations)
            if concentrations is not None:
                with self.cfg.lock:
                    self.cfg.concentrations = concentrations
            log.debug('poll_device: Sleeping for %d seconds.' % self.cfg.poll_interval)
            time.sleep(self.cfg.poll_interval)

class AQI(weewx.xtypes.XType):
    """
    AQI XType which computes the AQI (air quality index) from
    the pm2_5 value.
    """

    def __init__(self):
        pass

    agg_sql_dict = {
        'avg': "SELECT AVG(pm2_5), usUnits FROM %(table_name)s "
               "WHERE dateTime > %(start)s AND dateTime <= %(stop)s AND pm2_5 IS NOT NULL",
        'count': "SELECT COUNT(dateTime), usUnits FROM %(table_name)s "
                 "WHERE dateTime > %(start)s AND dateTime <= %(stop)s AND pm2_5 IS NOT NULL",
        'first': "SELECT pm2_5, usUnits FROM %(table_name)s "
                 "WHERE dateTime = (SELECT MIN(dateTime) FROM %(table_name)s "
                 "WHERE dateTime > %(start)s AND dateTime <= %(stop)s AND pm2_5 IS NOT NULL",
        'last': "SELECT pm2_5, usUnits FROM %(table_name)s "
                "WHERE dateTime = (SELECT MAX(dateTime) FROM %(table_name)s "
                "WHERE dateTime > %(start)s AND dateTime <= %(stop)s AND pm2_5 IS NOT NULL",
        'min': "SELECT pm2_5, usUnits FROM %(table_name)s "
               "WHERE dateTime > %(start)s AND dateTime <= %(stop)s AND pm2_5 IS NOT NULL "
               "ORDER BY pm2_5 ASC LIMIT 1;",
        'max': "SELECT pm2_5, usUnits FROM %(table_name)s "
               "WHERE dateTime > %(start)s AND dateTime <= %(stop)s AND pm2_5 IS NOT NULL "
               "ORDER BY pm2_5 DESC LIMIT 1;",
        'sum': "SELECT SUM(pm2_5), usUnits FROM %(table_name)s "
               "WHERE dateTime > %(start)s AND dateTime <= %(stop)s AND pm2_5 IS NOT NULL)",
    }

    @staticmethod
    def compute_pm2_5_aqi(pm2_5):
        #             U.S. EPA PM2.5 AQI
        #
        #  AQI Category  AQI Value  24-hr PM2.5
        # Good             0 -  50    0.0 -  12.0
        # Moderate        51 - 100   12.1 -  35.4
        # USG            101 - 150   35.5 -  55.4
        # Unhealthy      151 - 200   55.5 - 150.4
        # Very Unhealthy 201 - 300  150.5 - 250.4
        # Hazardous      301 - 400  250.5 - 350.4
        # Hazardous      401 - 500  350.5 - 500.4

        # The EPA standard for AQI says to truncate PM2.5 to one decimal place.
        # See https://www3.epa.gov/airnow/aqi-technical-assistance-document-sept2018.pdf
        x = math.trunc(pm2_5 * 10) / 10

        if x <= 12.0: # Good
            return x / 12.0 * 50
        elif x <= 35.4: # Moderate
            return (x - 12.1) / 23.3 * 49.0 + 51.0
        elif x <= 55.4: # Unhealthy for senstive
            return (x - 35.5) / 19.9 * 49.0 + 101.0
        elif x <= 150.4: # Unhealthy
            return (x - 55.5) / 94.9 * 49.0 + 151.0
        elif x <= 250.4: # Very Unhealthy
            return (x - 150.5) / 99.9 * 99.0 + 201.0
        elif x <= 350.4: # Hazardous
            return (x - 250.5) / 99.9 * 99.0 + 301.0
        else: # Hazardous
            return (x - 350.5) / 149.9 * 99.0 + 401.0

    @staticmethod
    def compute_pm2_5_aqi_color(pm2_5_aqi):
        if pm2_5_aqi <= 50:
            return 128 << 8                 # Green
        elif pm2_5_aqi <= 100:
            return (255 << 16) + (255 << 8) # Yellow
        elif pm2_5_aqi <=  150:
            return (255 << 16) + (140 << 8) # Orange
        elif pm2_5_aqi <= 200:
            return 255 << 16                # Red
        elif pm2_5_aqi <= 300:
            return (128 << 16) + 128        # Purple
        else:
            return 128 << 16                # Maroon

    @staticmethod
    def compute_pm2_5_us_epa_correction(pm2_5_cf_1: float, pm2_5_cf_1_b: float, current_humidity: int, current_temp_f: int) -> float:
        # PM2.5=0.541*PA_cf1(avgAB)-0.0618*RH +0.00534*T +3.634
        val = 0.541 * (pm2_5_cf_1 + pm2_5_cf_1_b) / 2.0 - 0.0618 * current_humidity + 0.00534 * current_temp_f + 3.634
        return val if val >= 0.0 else 0.0

    @staticmethod
    def get_scalar(obs_type, record, db_manager=None):
        log.debug('get_scalar(%s)' % obs_type)
        if obs_type not in [ 'pm2_5_aqi', 'pm2_5_aqi_color' ]:
            raise weewx.UnknownType(obs_type)
        log.debug('get_scalar(%s)' % obs_type)
        if record is None:
            log.debug('get_scalar called where record is None.')
            raise weewx.CannotCalculate(obs_type)
        if 'pm2_5' not in record:
            # Returning CannotCalculate causes exception in ImageGenerator, return UnknownType instead.
            # ERROR weewx.reportengine: Caught unrecoverable exception in generator 'weewx.imagegenerator.ImageGenerator'
            log.debug('get_scalar called where record does not contain pm2_5.')
            raise weewx.UnknownType(obs_type)
        if record['pm2_5'] is None:
            # Returning CannotCalculate causes exception in ImageGenerator, return UnknownType instead.
            # ERROR weewx.reportengine: Caught unrecoverable exception in generator 'weewx.imagegenerator.ImageGenerator'
            # This will happen for any catchup records inserted at weewx startup.
            log.debug('get_scalar called where record[pm2_5] is None.')
            raise weewx.UnknownType(obs_type)
        try:
            pm2_5 = record['pm2_5']
            if obs_type == 'pm2_5_aqi':
                value = AQI.compute_pm2_5_aqi(pm2_5)
            if obs_type == 'pm2_5_aqi_color':
                value = AQI.compute_pm2_5_aqi_color(AQI.compute_pm2_5_aqi(pm2_5))
            t, g = weewx.units.getStandardUnitType(record['usUnits'], obs_type)
            # Form the ValueTuple and return it:
            return weewx.units.ValueTuple(value, t, g)
        except KeyError:
            # Don't have everything we need. Raise an exception.
            raise weewx.CannotCalculate(obs_type)

    @staticmethod
    def get_series(obs_type, timespan, db_manager, aggregate_type=None, aggregate_interval=None):
        """Get a series, possibly with aggregation.
        """

        if obs_type not in [ 'pm2_5_aqi', 'pm2_5_aqi_color' ]:
            raise weewx.UnknownType(obs_type)

        log.debug('get_series(%s, %s, %s, aggregate:%s, aggregate_interval:%s)' % (
            obs_type, timestamp_to_string(timespan.start), timestamp_to_string(
            timespan.stop), aggregate_type, aggregate_interval))

        #  Prepare the lists that will hold the final results.
        start_vec = list()
        stop_vec = list()
        data_vec = list()

        # Is aggregation requested?
        if aggregate_type:
            # Yes. Just use the regular series function.
            return weewx.xtypes.ArchiveTable.get_series(obs_type, timespan, db_manager, aggregate_type,
                                           aggregate_interval)
        else:
            # No aggregation.
            sql_str = 'SELECT dateTime, usUnits, `interval`, pm2_5 FROM %s ' \
                      'WHERE dateTime >= ? AND dateTime <= ? AND pm2_5 IS NOT NULL' \
                      % db_manager.table_name
            std_unit_system = None

            for record in db_manager.genSql(sql_str, timespan):
                ts, unit_system, interval, pm2_5 = record
                if std_unit_system:
                    if std_unit_system != unit_system:
                        raise weewx.UnsupportedFeature(
                            "Unit type cannot change within a time interval.")
                else:
                    std_unit_system = unit_system

                if obs_type == 'pm2_5_aqi':
                    value = AQI.compute_pm2_5_aqi(pm2_5)
                if obs_type == 'pm2_5_aqi_color':
                    value = AQI.compute_pm2_5_aqi_color(AQI.compute_pm2_5_aqi(pm2_5))
                log.debug('get_series(%s): %s - %s - %s' % (obs_type,
                    timestamp_to_string(ts - interval * 60),
                    timestamp_to_string(ts), value))
                start_vec.append(ts - interval * 60)
                stop_vec.append(ts)
                data_vec.append(value)

            unit, unit_group = weewx.units.getStandardUnitType(std_unit_system, obs_type,
                                                               aggregate_type)

        return (ValueTuple(start_vec, 'unix_epoch', 'group_time'),
                ValueTuple(stop_vec, 'unix_epoch', 'group_time'),
                ValueTuple(data_vec, unit, unit_group))

    @staticmethod
    def get_aggregate(obs_type, timespan, aggregate_type, db_manager, **option_dict):
        """Returns an aggregation of pm2_5_aqi over a timespan by using the main archive
        table.

        obs_type: Must be 'pm2_5_aqi' or 'pm2_5_aqi_color'.

        timespan: An instance of weeutil.Timespan with the time period over which aggregation is to
        be done.

        aggregate_type: The type of aggregation to be done. For this function, must be 'avg',
        'sum', 'count', 'first', 'last', 'min', or 'max'. Anything else will cause
        weewx.UnknownAggregation to be raised.

        db_manager: An instance of weewx.manager.Manager or subclass.

        option_dict: Not used in this version.

        returns: A ValueTuple containing the result.
        """
        if obs_type not in [ 'pm2_5_aqi', 'pm2_5_aqi_color' ]:
            raise weewx.UnknownType(obs_type)

        log.debug('get_aggregate(%s, %s, %s, aggregate:%s)' % (
            obs_type, timestamp_to_string(timespan.start),
            timestamp_to_string(timespan.stop), aggregate_type))

        aggregate_type = aggregate_type.lower()

        # Raise exception if we don't know about this type of aggregation
        if aggregate_type not in list(AQI.agg_sql_dict.keys()):
            raise weewx.UnknownAggregation(aggregate_type)

        # Form the interpolation dictionary
        interpolation_dict = {
            'start': timespan.start,
            'stop': timespan.stop,
            'table_name': db_manager.table_name
        }

        select_stmt = AQI.agg_sql_dict[aggregate_type] % interpolation_dict
        row = db_manager.getSql(select_stmt)
        if row:
            value, std_unit_system = row
        else:
            value = None
            std_unit_system = None

        if value is not None:
            if obs_type == 'pm2_5_aqi':
                value = AQI.compute_pm2_5_aqi(value)
            if obs_type == 'pm2_5_aqi_color':
                value = AQI.compute_pm2_5_aqi_color(AQI.compute_pm2_5_aqi(value))
        t, g = weewx.units.getStandardUnitType(std_unit_system, obs_type, aggregate_type)
        # Form the ValueTuple and return it:
        log.debug('get_aggregate(%s, %s, %s, aggregate:%s, select_stmt: %s, returning %s)' % (
            obs_type, timestamp_to_string(timespan.start), timestamp_to_string(timespan.stop),
            aggregate_type, select_stmt, value))
        return weewx.units.ValueTuple(value, t, g)

if __name__ == "__main__":
    usage = """%prog [options] [--help] [--debug]"""

    import weeutil.logger

    def main():
        import optparse
        parser = optparse.OptionParser(usage=usage)
        parser.add_option('--config', dest='cfgfn', type=str, metavar="FILE",
                          help="Use configuration file FILE. Default is /etc/weewx/weewx.conf or /home/weewx/weewx.conf")
        parser.add_option('--test-collector', dest='tc', action='store_true',
                          help='test the data collector')
        parser.add_option('--test-is-sane', dest='sane_test', action='store_true',
                          help='test the is_sane function')
        parser.add_option('--sensor-id', dest='sensor_id', action='store',
                          help='sensor id to use with --test-collector')
        (options, args) = parser.parse_args()

        weeutil.logger.setup('purple', {})

        if options.tc:
            if not options.sensor_id:
                parser.error('--test-collector requires --sensor-id argument')
            test_collector(options.sensor_id)
        if options.sane_test:
            test_is_sane()

    def test_collector(sensor_id):
        while True:
            print(collect_data(sensor_id, 10, 300))
            time.sleep(5)

    def test_is_sane():
        good_device = ('{"SensorId":"84:f3:eb:36:38:fe","DateTime":"2020/03/20T17:18:02z",'
            '"Geo":"PurpleAir-38fe","Mem":19176,"memfrag":15,"memfb":16360,"memcs":768,'
            '"Id":16220,"lat":37.431599,"lon":-122.111000,"Adc":0.03,"loggingrate":15,'
            '"place":"outside","version":"6.01","uptime":215685,"rssi":-59,"period":120,'
            '"httpsuccess":10842,"httpsends":10842,"hardwareversion":"2.0",'
            '"hardwarediscovered":"2.0+OPENLOG+NO-DISK+DS3231+BME280+PMSX003-B+PMSX003-A",'
            '"current_temp_f":61,"current_humidity":48,"current_dewpoint_f":41,'
            '"pressure":1024.30,"p25aqic_b":"rgb(4,228,0)","pm2.5_aqi_b":12,'
            '"pm1_0_cf_1_b":1.63,"p_0_3_um_b":556.21,"pm2_5_cf_1_b":2.95,'
            '"p_0_5_um_b":150.61,"pm10_0_cf_1_b":3.25,"p_1_0_um_b":22.58,'
            '"pm1_0_atm_b":1.63,"p_2_5_um_b":2.11,"pm2_5_atm_b":2.95,"p_5_0_um_b":0.46,'
            '"pm10_0_atm_b":3.25,"p_10_0_um_b":0.26,"p25aqic":"rgb(10,229,0)",'
            '"pm2.5_aqi":17,"pm1_0_cf_1":2.20,"p_0_3_um":637.30,"pm2_5_cf_1":4.02,'
            '"p_0_5_um":174.22,"pm10_0_cf_1":4.43,"p_1_0_um":28.53,"pm1_0_atm":2.20,'
            '"p_2_5_um":3.97,"pm2_5_atm":4.02,"p_5_0_um":0.50,"pm10_0_atm":4.43,'
            '"p_10_0_um":0.50,"pa_latency":338,"response":201,"response_date":1584724649,'
            '"latency":355,"key1_response":200,"key1_response_date":1584724642,'
            '"key1_count":81455,"ts_latency":805,"key2_response":200,'
            '"key2_response_date":1584724644,"key2_count":81455,"ts_s_latency":796,'
            '"key1_response_b":200,"key1_response_date_b":1584724645,"key1_count_b":81444,'
            '"ts_latency_b":772,"key2_response_b":200,"key2_response_date_b":1584724647,'
            '"key2_count_b":81446,"ts_s_latency_b":796,"wlstate":"Connected","status_0":2,'
            '"status_1":2,"status_2":2,"status_3":2,"status_4":2,"status_5":2,"status_6":2,'
            '"status_7":0,"status_8":2,"status_9":2,"ssid":"ella"}')
        bad_1 = ('{"SensorId":"84:f3:eb:36:38:fe","DateTime":"2020/03/18T05:23:59z",'
            ' "current_temp_f":53, "current_humidity":57, "current_dewpoint_f":38,'
            ' "pressure":1015.94, "pm1_0_cf_1":"nan", "pm1_0_atm":"nan", "p_0_3_um":"nan",'
            ' "pm2_5_cf_1":"nan", "pm2_5_atm":"nan", "p_0_5_um":"nan", "pm10_0_cf_1":"nan",'
            ' "pm10_0_atm":"nan", "pm2.5_aqi":"nan", "p25aqic":"rgb(0,255,255)",'
            ' "pm1_0_cf_1_b":"nan", "pm1_0_atm_b":"nan", "p_0_3_um_b":"nan",'
            ' "pm2_5_cf_1_b":"nan", "pm2_5_atm_b":"nan", "p_0_5_um_b":"nan",'
            ' "pm10_0_cf_1_b":"nan", "pm10_0_atm_b":"nan",'
            ' "pm2_5_aqi_b":"nan", "p25aqic_b":"rgb(0,255,255)"}')
        j = json.loads(good_device)
        assert(is_sane(j))
        j = json.loads(bad_1)
        assert(not is_sane(j))

    main()
