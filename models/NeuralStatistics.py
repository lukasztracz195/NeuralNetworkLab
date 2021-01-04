class NeuralStatistic:

    def __init__(self):
        self.__condensed_error = list()
        self.__error_for_series = list()
        self.__error_for_epocs = list()
        self.__history_metrics = dict()

    def clear(self):
        self.__condensed_error = list()
        self.__error_for_series = list()
        self.__error_for_epocs = list()
        self.__history_metrics = dict()

    @property
    def condensed_error(self):
        return self.__condensed_error

    @condensed_error.setter
    def condensed_error(self, value):
        self.__condensed_error = value

    @property
    def error_for_series(self):
        return self.__error_for_series

    @error_for_series.setter
    def error_for_series(self, value):
        self.__error_for_series = value

    @property
    def error_for_epocs(self):
        return self.__error_for_epocs

    @error_for_epocs.setter
    def error_for_epocs(self, value):
        self.__error_for_epocs = value

    @property
    def history_metrics(self):
        return self.__history_metrics

    @history_metrics.setter
    def history_metrics(self, value):
        self.__history_metrics = value
