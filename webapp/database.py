class DB:
    def __init__(self):
        """
        Simple in-memory DB replacement for logging IP and email pairs.
        """
        self.data = {}

    def set(self, ip, email):
        self.data[ip] = email

    def get(self, ip):
        return self.data.get(ip, None)
