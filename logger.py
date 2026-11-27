import csv

class EventLogger:
    def __init__(self,path):
        self.file=open(path,'w',newline='',encoding='utf-8')
        self.writer=csv.DictWriter(self.file,fieldnames=['timestamp_s','event','source','details'])
        self.writer.writeheader()
    def log(self,timestamp_s,event,source,details=''):
        self.writer.writerow(dict(timestamp_s=round(timestamp_s,3),event=event,source=source,details=details)); self.file.flush()
    def close(self): self.file.close()
