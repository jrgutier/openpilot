from opendbc.car.rivian.fingerprints import FW_VERSIONS
from opendbc.car.rivian.values import CAR, FW_QUERY_CONFIG, WMI, ModelYear, narrow_rivian_fw_match_by_vin
from opendbc.car.fw_versions import match_fw_to_car
from opendbc.car.car_helpers import normalize_car_fingerprint


class TestRivian:
  def test_vin_fuzzy_wmi_year_r1t(self):
    vin = "7FCABCD12N1234567"
    assert FW_QUERY_CONFIG.match_fw_to_car_fuzzy({}, vin, FW_VERSIONS) == {str(CAR.RIVIAN_R1T_GEN1)}

  def test_vin_fuzzy_wmi_year_r1s(self):
    vin = "7PDABCD12N1234567"
    assert FW_QUERY_CONFIG.match_fw_to_car_fuzzy({}, vin, FW_VERSIONS) == {str(CAR.RIVIAN_R1S_GEN1)}

  def test_vin_fuzzy_unknown_year(self):
    vin = "7FCABCD12Z1234567"
    assert FW_QUERY_CONFIG.match_fw_to_car_fuzzy({}, vin, FW_VERSIONS) == set()

  def test_narrow_rivian_after_shared_fw_match(self):
    r1t, r1s = str(CAR.RIVIAN_R1T_GEN1), str(CAR.RIVIAN_R1S_GEN1)
    assert narrow_rivian_fw_match_by_vin({r1t, r1s}, "7PDABCD12R1234567") == {r1s}
    assert narrow_rivian_fw_match_by_vin({r1t, r1s}, "7FCABCD12R1234567") == {r1t}

  def test_full_match_fw_to_car_empty_fw_vin_only(self):
    vin = "7PDABCD12R1234567"
    exact, matches = match_fw_to_car([], vin)
    assert exact is False
    assert matches == {str(CAR.RIVIAN_R1S_GEN1)}

  def test_normalize_deprecated_platform(self):
    assert normalize_car_fingerprint("RIVIAN_R1_GEN1") == str(CAR.RIVIAN_R1S_GEN1)
