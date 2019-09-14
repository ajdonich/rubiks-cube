import numpy as np

class Observer:
#{
    def sm_move_notification(self, rot_mats, mindex, alpha_masks): pass

    # def notify(self, observable, *args, **kwargs):
    #     print('Got', args, kwargs, 'From', observable)
#}

class Observable:
#{
    def __init__(self):
        self._observers = []
    
    def register_observer(self, observer):
        self._observers.append(observer)
    
    def unregister_observer(self, observer):
        self._observers.remove(observer)

    def notify_observers(self, rot_mats, mindex, alpha_masks=None):
        for observer in self._observers:
            observer.sm_move_notification(rot_mats, mindex, alpha_masks)

    # def notify_observers(self, *args, **kwargs):
    #     for observer in self._observers:
    #         observer.notify(self, *args, **kwargs)
#}
