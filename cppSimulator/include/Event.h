#ifndef __EVENT_H__
#define __EVENT_H__

#include <functional>
#include <queue>
#include <memory>

//forward decl
class Sprite;

class Event {
public:
    Event(double time, std::function<void(void)> fn):
    time(time),fn(fn){}
    inline void activate() {fn();}
    inline bool operator < (const Event& other) const {
        return !(time < other.time);
    }
    inline double get_time() const {
        return time;
    }

    ~Event(){}
private:
    double time;
    std::function<void(void)> fn;
};

class EventFactory{
public:
    static void CreateAttackEvnt(Sprite* attacker,
        Sprite* victim);
    static void CreateSpawnEvnt(cppSimulatorImp* Engine);
};

#endif//__EVENT_H__