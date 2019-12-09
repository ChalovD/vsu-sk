from unittest import TestCase

from main.schrodinger_equation_builder import Schrodinger_equation_builder
from main.schrodinger_equation_solver import Schrodinger_equation_solver


class TestSchrodinger_equation_builder(TestCase):
    def test_building(self):
        potential = lambda x: x * x
        matrix_size = 501
        left_boundary = -10
        right_boundary = 10

        just_solver = Schrodinger_equation_solver(potential, matrix_size, left_boundary, right_boundary)
        built_solver = Schrodinger_equation_builder() \
            .set_point_number(matrix_size) \
            .set_boundaries(left_boundary, right_boundary) \
            .set_potential(potential) \
            .build()

        self.assertEqual(just_solver.start_point, built_solver.start_point)
        self.assertEqual(just_solver.end_point, built_solver.end_point)
        self.assertEqual(just_solver.approach_matrix_size, built_solver.approach_matrix_size)
        self.assertEqual(just_solver.potential, built_solver.potential)