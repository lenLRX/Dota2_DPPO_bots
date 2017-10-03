#ifndef __EVENT_H__
#define __EVENT_H__

#include <functional>
#include <queue>

class Event {
public:
    Event(double time, std::function<void(void)> fn):
    time(time),fn(fn){}
    inline void activate() {fn();}
    inline bool operator < (const Event& other) const {
        return time < other.time;
    }
    inline double get_time() const {
        return time;
    }
private:
    double time;
    std::function<void(void)> fn;
};

#endif//__EVENT_H__