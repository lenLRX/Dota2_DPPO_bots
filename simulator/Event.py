import heapq

class Event(object):
    def __init__(self, time, callback, args):
        self.time = time
        self.callback = callback
        self.args = args
    
    def activate(self):
        self.callback(*self.args)
        
    def __lt__(self, other):
        return self.time < other.time

    def __cmp__(self, other):
        return self.time - other.time


class EventQueue(object):
    def __init__(self):
        self._heap = []
    
    def enqueue(self, event):
        heapq.heappush(self._heap, event)
    
    def fetch(self, tick_time):
        if len(self._heap) == 0:
            return None

        if self._heap[0].time <= tick_time:
            ret = self._heap[0]
            heapq.heappop(self._heap)
            return ret