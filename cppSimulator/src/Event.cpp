#include "simulatorImp.h"
#include "Event.h"
#include "Sprite.h"

#include "Creep.h"

void EventFactory::CreateAttackEvnt(Sprite* attacker, Sprite* victim)
{
    cppSimulatorImp* engine = attacker->get_engine();
    auto canvas = engine->get_canvas();
    PyObject* vhandle = nullptr;
    if (canvas)
    {
        auto p_attacker = attacker->pos_in_wnd();
        auto p_victim = victim->pos_in_wnd();
        auto fn = PyObject_GetAttrString(canvas, "create_line");
        auto args = Py_BuildValue("(dddd)",
            std::get<0>(p_attacker),
            std::get<1>(p_attacker),
            std::get<0>(p_victim),
            std::get<1>(p_victim));
        auto kw = Py_BuildValue("{s:s}","arrow","last");
        vhandle = PyObject_Call(fn, args, kw);
        Py_XDECREF(fn);
        Py_XDECREF(args);
        Py_XDECREF(kw);
    }
    auto fn = [=]() {
        victim->damadged(attacker, attacker->get_Attack());
        if (vhandle)
        {
            PyObject* delete_fn = PyObject_GetAttrString(canvas, "delete");
            PyObject* args = Py_BuildValue("(O)", vhandle);
            PyObject* kwargs = Py_BuildValue("{}");
            Py_XDECREF(PyObject_Call(delete_fn, args, kwargs));
            Py_DECREF(kwargs);
            Py_DECREF(args);
            Py_DECREF(delete_fn);
            Py_DECREF(vhandle);
        }
    };
    engine->get_queue().push(
        Event(engine->get_time() + attacker->TimeToDamage(victim), fn));
}

static void spawn_fn(cppSimulatorImp* Engine) {
    for (int i = 0; i < 5; ++i) {
        Engine->addSprite(new Creep(Engine, Side::Radiant, "MeleeCreep"));
        Engine->addSprite(new Creep(Engine, Side::Dire, "MeleeCreep"));
    }

    Engine->addSprite(new Creep(Engine, Side::Radiant, "RangedCreep"));
    Engine->addSprite(new Creep(Engine, Side::Dire, "RangedCreep"));
    
    std::function<void()> fn = std::bind(spawn_fn, Engine);
    Engine->get_queue().push(Event(Engine->get_time() + 30, fn));
}

void EventFactory::CreateSpawnEvnt(cppSimulatorImp* Engine)
{
    std::function<void()> fn = std::bind(spawn_fn, Engine);
    Engine->get_queue().push(Event(Engine->get_time() + 90,fn));
}
