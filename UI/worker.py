from PySide6.QtCore import QThread, Signal

# create worker object to send job and job data
class HPCWorker(QThread):
    # signals for GUI to listen for
    finished = Signal(dict)  # signal when job finishes successfully
    error = Signal(str)      # signal when error

    def __init__(self, job_data, send_job_func):
        super().__init__()
        self.job_data = job_data          # prompt, model, watermark, etc.
        self.send_job_func = send_job_func  # see send_job_to_hpc() in app-hpc.py

    def run(self):
        #run in separate theread
        try:
            result = self.send_job_func(self.job_data)
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))
