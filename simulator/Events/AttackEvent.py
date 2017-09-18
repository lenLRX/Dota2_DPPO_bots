from ..Event import Event

class AttackEvent(object):
    @staticmethod
    def Create(attacker, victim):
        _engine = attacker.Engine
        _engine.event_queue.enqueue(Event(_engine.get_time() + attacker.AttackTime,
        AttackEvent._attack, (attacker,victim)))
    
    @staticmethod
    def _attack(attacker, victim):
        _engine = attacker.Engine
        victim.damadged(attacker.Attack)