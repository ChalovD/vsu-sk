from matplotlib import pyplot
from matplotlib.ticker import NullFormatter
from numpy import core, diag, linspace
from numpy.linalg import eigh, eigvalsh

class Schrodinger_equation_solver:
    potential = lambda: print("Potential function invoked but didn't set.")  # From double to double lambda
    approach_matrix_size = 0  # Int variable
    start_point = 0  # Double variable
    end_point = 0  # Double variable

    # Private fields
    _kinetic_matrix = None  # Square numpy array
    _potential_matrix = None  # Square numpy array

    _eig_energies = None  #
    _eig_functions = None  #

    def __init__(self, potential, approach_matrix_size, start_point, end_point):
        self.potential = potential
        self.approach_matrix_size = approach_matrix_size
        self.start_point = start_point
        self.end_point = end_point

        self.prepare_kinetic_matrix(approach_matrix_size)
        self.prepare_potential_matrix(potential)

    def solve(self):
        self._eig_energies, self._eig_functions = eigh(self._kinetic_matrix + self._potential_matrix)

    def print_energies(self, start, end):
        figure, axe = pyplot.subplots()
        figure.suptitle("Energy spectrum in the stationary state")

        x_points = linspace(start, end, end - start + 1)
        y_points = self._eig_energies[start - 1: end]

        axe.plot(x_points, y_points, 'ro')
        pyplot.xlabel("Number")
        pyplot.ylabel("Energy")
        pyplot.show()

    def print_wave_functions(self, numbers):
        x_points = linspace(self.start_point, self.end_point, self.approach_matrix_size)
        y_points = lambda n: self._eig_functions[:, n - 1]

        figure = pyplot.figure()

        for number in numbers:
            axe = figure.add_subplot(len(numbers), 1, number)
            axe.plot(x_points, y_points(number))
            axe.set_title("Wave function that's number is %s" % number)
            axe.set_xlabel('Configuration space variable')
            axe.set_ylabel('Wave function')

        pyplot.gca().yaxis.set_minor_formatter(NullFormatter())
        pyplot.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=1, wspace=0.35)
        pyplot.show()

    def print_probability_density_function(self, numbers):
        x_points = linspace(self.start_point, self.end_point, self.approach_matrix_size)
        y_points = lambda n: self._eig_functions[:, n - 1] ** 2

        figure = pyplot.figure()

        for number in numbers:
            axe = figure.add_subplot(len(numbers), 1, number)
            axe.plot(x_points, y_points(number))
            axe.set_title("Probability density function that's number is %s" % number)
            axe.set_xlabel('Configuration space variable')
            axe.set_ylabel('Probability density function')

        pyplot.gca().yaxis.set_minor_formatter(NullFormatter())
        pyplot.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=1, wspace=0.35)
        pyplot.show()

    # Private methods
    def prepare_kinetic_matrix(self, approach_matrix_size):
        # Dirty using side effect of fill_diagonal
        number_of_interval = self.approach_matrix_size - 1
        step = abs(self.start_point - self.end_point) / number_of_interval

        self._kinetic_matrix = core.zeros((approach_matrix_size, approach_matrix_size))
        self.fill_diagonal(self._kinetic_matrix, -1, -0.5 / (step ** 2))
        self.fill_diagonal(self._kinetic_matrix, 0, 1. / (step ** 2))
        self.fill_diagonal(self._kinetic_matrix, 1, -0.5 / (step ** 2))

    def prepare_potential_matrix(self, potential):
        grid = linspace(self.start_point, self.end_point, self.approach_matrix_size)
        self._potential_matrix = diag(potential(grid))

    @staticmethod
    def fill_diagonal(matrix, diagonal_position, filled_value):
        """
        Parameters:
            matrix (ndarray): The matrix with (size, size) format shape that will be filled by diagonals.
            diagonal_position (int): The diagonal_position member number of column where diagonal presented in first row if it positive and number of row where diagonal presented in the first column if it negative.
            filled_value (double): The filled_value will be filled to the diagonal_position diagonal.

        Returns:
            Matrix (ndarray): matrix with filled diagonal.

        Note:
            To respect performance reason function returns the matrix that passed by argument. Be careful,
            don't use passed argument.
        """
        shape = matrix.shape
        if (not (len(shape) == 2)) or (not (shape[0] == shape[1])):
            raise Exception("Matrix {} must be square.", matrix)

        size = shape[0]
        if abs(diagonal_position) > size:
            raise Exception("{} must be no more then {}.", diagonal_position, size)

        if diagonal_position >= 0:
            for index in range(diagonal_position, size):
                difference = index - abs(diagonal_position)
                matrix[difference, index] = filled_value
        else:
            for index in range(abs(diagonal_position), size):
                difference = index - abs(diagonal_position)
                matrix[index, difference] = filled_value
        return matrix
