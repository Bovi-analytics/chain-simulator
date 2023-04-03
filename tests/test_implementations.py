from scipy.sparse import csr_array

from chain_simulator.implementations import ScipyCSRAssembler, chain_simulator


class TestScipyCSRAssembler:
    def test_state_combinations(self):
        states = ("1", "2", "3")
        combinations = list(ScipyCSRAssembler.state_combinations(states))
        expected = [
            ("1", "1"),
            ("1", "2"),
            ("1", "3"),
            ("2", "1"),
            ("2", "2"),
            ("2", "3"),
            ("3", "1"),
            ("3", "2"),
            ("3", "3"),
        ]
        assert combinations == expected

    def test_allocate_array(self):
        coo_array = ScipyCSRAssembler.allocate_array(50)
        assert coo_array.format == "lil"
        assert coo_array.dtype == "float64"
        assert coo_array.shape == (50, 50)

    def test_states_to_index(self):
        states = ("A", "B", "C")
        index = ScipyCSRAssembler.states_to_index(states)
        expected = {
            "A": 0,
            "B": 1,
            "C": 2
        }
        assert index == expected


class TestChainSimulator:
    def test_matmul_1(self):
        array = csr_array([
            [0.0, 1.0, 0.0],
            [0.0, 0.5, 0.5],
            [0.0, 0.0, 1.0]
        ])
        result = chain_simulator(array, 1)
        expected = csr_array([
            [0.00, 0.50, 0.50],
            [0.00, 0.25, 0.75],
            [0.00, 0.00, 1.00]
        ])
        comparison = result == expected
        assert len(comparison.data) == 9

    def test_matmul_2(self):
        array = csr_array([
            [0.0, 1.0, 0.0],
            [0.0, 0.5, 0.5],
            [0.0, 0.0, 1.0]
        ])
        result = chain_simulator(array, 2)
        expected = csr_array([
            [0.000, 0.250, 0.750],
            [0.000, 0.125, 0.875],
            [0.000, 0.000, 1.000]
        ])
        comparison = result == expected
        # print(result.toarray())
        print(type(expected.sum(1)))
        assert len(comparison.data) == 9
