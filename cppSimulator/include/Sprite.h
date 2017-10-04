#ifndef __SPRITE_H__
#define __SPRITE_H__

#include <Python.h>
#include "Config.h"

//forward decl
class cppSimulatorImp;

class Sprite {
public:
    Sprite(cppSimulatorImp* Engine,
        PyObject* canvas,
        Side side, pos_tup loc, double HP,
        double MP, double Speed,double Armor,
        double ATK,double ATKRange,double SightRange,
        double Bounty,double bountyEXP,
        double BAT,double AS):
        Engine(Engine),canvas(canvas),side(side),
        location(loc),HP(HP),MP(MP),MovementSpeed(Speed),
        BaseAttackTime(BAT),AttackSpeed(AS),Armor(Armor),
        Attack(ATK),AttackRange(ATKRange),SightRange(SightRange),
        Bounty(Bounty), bountyEXP(bountyEXP), LastAttackTime(-1),
        exp(0),isDead(false),b_move(false), v_handle(NULL)
    {   
        _update_para();
    }

    virtual ~Sprite(){}

    inline void _update_para() {
        double AttackPerSecond = AttackSpeed * 0.01 / BaseAttackTime;
        AttackTime = 1 / AttackPerSecond;
    }

    virtual void step() = 0;
    inline pos_tup pos_in_wnd() {
        return pos_tup(std::get<0>(location) * Config::game2window_scale * 0.5 + Config::windows_size * 0.5,
            std::get<1>(location) * Config::game2window_scale * 0.5 + Config::windows_size * 0.5);
    }

    void attack(const Sprite& target);
    bool isAttacking();
    inline void set_move(pos_tup target) {
        move_target = target;
    }
    void move();
    void damadged();
    void dead();

    static double S2Sdistance(const Sprite& s1,const Sprite& s2);

protected:
    cppSimulatorImp* Engine;
    PyObject* canvas;
    Side side;
    pos_tup location;
    double HP;
    double MP;
    double MovementSpeed;
    double BaseAttackTime;
    double AttackSpeed;
    double Armor;
    double Attack;
    double AttackRange;
    double SightRange;
    double Bounty;
    double bountyEXP;
    double LastAttackTime;
    double AttackTime;
    double exp;
    bool isDead;
    bool b_move;
    PyObject* v_handle;
    pos_tup move_target;
};

#endif//__SPRITE_H__