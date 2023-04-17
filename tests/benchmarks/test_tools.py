from benchmarks.array_assembly_method import name_timings


class TestNameTimings:
    def test_zero_timings(self):
        """Test namings with zero timings."""
        timings = []
        named = name_timings(timings)
        expected = {}
        assert named == expected

    def test_one_timing(self):
        """Test namings with one timing."""
        timings = [1]
        named = name_timings(timings)
        expected = {"run_00": 1}
        assert named == expected

    def test_three_timings(self):
        """Test namings with three timings."""
        timings = [3, 1, 2]
        named = name_timings(timings)
        expected = {"run_00": 3, "run_01": 1, "run_02": 2}
        assert named == expected

    def test_timing_int(self):
        """Test type of timings is `int`."""
        timings = [1]
        named = name_timings(timings)
        assert isinstance(named["run_00"], int)

    def test_timing_type_float(self):
        """Test type of timings is `float`."""
        timings = [1.0]
        named = name_timings(timings)
        assert isinstance(named["run_00"], float)
