#include "simulatorImp.h"
#include "Event.h"
#include "Sprite.h"

#include "Creep.h"

void EventFactory::CreateAttackEvnt(Sprite* attacker, Sprite* victim)
{
    cppSimulatorImp* engine = attacker->get_engine();
    auto fn = [=]() {
        victim->damadged(attacker->get_Attack());
    };
    engine->get_queue().push(
        Event(engine->get_time() + attacker->get_AttackTime(), fn));
}

static void spawn_fn(cppSimulatorImp* Engine) {
    for (int i = 0; i < 5; ++i) {
        Engine->addSprite(new Creep(Engine, Side::Radiant, "MeleeCreep"));
        Engine->addSprite(new Creep(Engine, Side::Dire, "MeleeCreep"));
    }
    
    std::function<void()> fn = std::bind(spawn_fn, Engine);
    Engine->get_queue().push(Event(Engine->get_time() + 30, fn));
}

void EventFactory::CreateSpawnEvnt(cppSimulatorImp* Engine)
{
    std::function<void()> fn = std::bind(spawn_fn, Engine);
    Engine->get_queue().push(Event(Engine->get_time() + 30,fn));
}
