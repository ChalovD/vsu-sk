from unittest import TestCase

from numpy import core, array_equal
from numpy import fill_diagonal
from numpy.ma import exp

from main.schrodinger_equation_solver import Schrodinger_equation_solver


class Test_schrodinger_equation_solver(TestCase):
    def test_fill_primary_diagonal(self):
        size = 10
        filled = 7

        zero = core.zeros((size, size))
        must_be = zero.copy()
        fill_diagonal(must_be, filled)

        current = Schrodinger_equation_solver.fill_diagonal(
            matrix=zero,
            diagonal_position=0,
            filled_value=filled
        )
        bul = current == must_be

        self.assertTrue(array_equal(must_be, current))

    def test_fill_side_diagonal(self):
        filled = 1.

        down_diagonal = [[0., 0.], [filled, 0.]]
        up_diagonal = [[0., filled], [0., 0.]]

        staple = core.zeros((2, 2))
        counted_down_diagonal = Schrodinger_equation_solver.fill_diagonal(
            matrix=staple.copy(),
            diagonal_position=-1,
            filled_value=filled
        )
        counted_up_diagonal = Schrodinger_equation_solver.fill_diagonal(
            matrix=staple.copy(),
            diagonal_position=1,
            filled_value=filled
        )

        self.assertTrue(array_equal(down_diagonal, counted_down_diagonal))
        self.assertTrue(array_equal(up_diagonal, counted_up_diagonal))

    def test_solve_for_oscillator(self):
        potential = lambda x: 0.5 * x * x
        matrix_size = 500
        left_boundary = -10
        right_boundary = 10

        solver = Schrodinger_equation_solver(potential, matrix_size, left_boundary, right_boundary)
        solver.solve()

    def test_print_solved(self):
        potential = lambda x: 2 * ( exp(-2 * x) - 2 * exp(-x) + 1 ) # 0.5 * x ** 2 + 0.5 * x ** 4
        matrix_size = 501
        left_boundary = -10
        right_boundary = 10

        solver = Schrodinger_equation_solver(potential, matrix_size, left_boundary, right_boundary)
        solver.solve()
        solver.print_energies(1, 10)
        solver.print_wave_functions((1, 2, 3, 4, 5))
        solver.print_probability_density_function((1, 2, 3, 4, 5))
