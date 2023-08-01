import time


class Timer(object):

    last_time = None
    start_time = None
    time_measures = []

    @staticmethod
    def start_timer():
        Timer.start_time = time.perf_counter()
        Timer.last_time = time.perf_counter()
        Timer.time_measures = []

    @staticmethod
    def log_time(task_name: str):
        current_time = time.perf_counter()
        task_time = current_time - Timer.last_time
        Timer.time_measures.append((task_name, round(task_time, 4)))
        Timer.last_time = current_time

    @staticmethod
    def print_task_times():
        current_time = time.perf_counter()
        complete_time = round(current_time - Timer.start_time, 4)

        hours = complete_time // 3600
        minutes = (complete_time % 3600) // 60
        remaining_seconds = (complete_time % 3600) % 60

        print(f"Complete execution time: \t {hours} hours, {minutes} minutes, and {remaining_seconds} seconds")

        for task_name, task_time in Timer.time_measures:
            hours = task_time // 3600
            minutes = (task_time % 3600) // 60
            remaining_seconds = (task_time % 3600) % 60

            print(f"{task_name}: \t {hours} hours, {minutes} minutes, and {remaining_seconds} seconds")


