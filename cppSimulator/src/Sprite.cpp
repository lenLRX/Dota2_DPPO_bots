#include "Sprite.h"
#include "simulatorImp.h"

void Sprite::attack(const Sprite& target)
{
    LastAttackTime = Engine->get_time();
}

bool Sprite::isAttacking()
{
    return (Engine->get_time() - LastAttackTime)
        < AttackTime;
}

void Sprite::move()
{
}

void Sprite::damadged()
{
}

void Sprite::dead()
{
}

double Sprite::S2Sdistance(const Sprite & s1, const Sprite & s2)
{
    return 0.0;
}
