from main.schrodinger_equation_solver import Schrodinger_equation_solver


class Schrodinger_equation_builder:
    _schrodinger_equation_solver = None
    _point_number = None
    _start_point = None
    _end_point = None
    _potential = None

    def set_point_number(self, point_number):
        self._point_number = point_number
        return self

    def set_boundaries(self, start_point, end_point):
        self._start_point = start_point
        self._end_point = end_point
        return self

    def set_potential(self, potential):
        self._potential = potential
        return self

    def build(self):
        return Schrodinger_equation_solver(
            potential=self._potential,
            approach_matrix_size=self._point_number,
            start_point=self._start_point,
            end_point=self._end_point
        )