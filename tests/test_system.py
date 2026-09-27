from ailab.core import system


def test_probe_never_crashes_and_reports_basics():
    info = system.probe()
    assert info.cpu_threads >= 1
    assert info.ram_total_mb > 0
    report = system.format_report(info)
    assert "CPU" in report and "Compute" in report


def test_windows_hardware_query_is_parsed():
    """The Windows probe's PowerShell JSON (captured shape), parsed on any OS."""
    text = ('{"cpu":{"Name":"AMD Ryzen 7 5800H with Radeon Graphics","NumberOfCores":8},'
            '"gpus":[{"Name":"NVIDIA GeForce RTX 3070 Laptop GPU","AdapterCompatibility":'
            '"NVIDIA","DriverVersion":"32.0.15.6094","AdapterRAM":4293918720},'
            '{"Name":"AMD Radeon(TM) Graphics","AdapterCompatibility":"Advanced Micro Devices,'
            ' Inc.","DriverVersion":"31.0.21001.45002","AdapterRAM":536870912},'
            '{"Name":"Microsoft Basic Display Adapter","AdapterCompatibility":"(Standard)"}]}')
    model, cores, gpus = system.parse_windows_query(text)
    assert model == "AMD Ryzen 7 5800H with Radeon Graphics" and cores == 8
    assert [g.vendor for g in gpus] == ["NVIDIA", "AMD"]
    assert gpus[0].name == "GeForce RTX 3070 Laptop GPU"


def test_windows_query_single_gpu_and_garbage():
    one = '{"cpu":{"Name":"Intel(R) Core(TM) i5","NumberOfCores":4},"gpus":{"Name":"Intel(R) UHD"}}'
    model, cores, gpus = system.parse_windows_query(one)
    assert model == "Intel Core i5" and len(gpus) == 1 and gpus[0].vendor == "Intel"
    assert system.parse_windows_query("not json") == ("", 0, [])
