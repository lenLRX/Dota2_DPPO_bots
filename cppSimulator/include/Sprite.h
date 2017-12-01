#ifndef __SPRITE_H__
#define __SPRITE_H__

#include <Python.h>
#include "Config.h"

#include <memory>
#include <unordered_map>
#include <vector>

//forward decl
class cppSimulatorImp;

enum AtkType
{
    melee,
    ranged
};

#define SETATTR(data,type,attr) attr = *(type*)data.at(#attr)

#define INIT_ATTR_BY(data)\
SETATTR(data, double, HP);\
SETATTR(data, double, MP);\
SETATTR(data, double, MovementSpeed);\
SETATTR(data, double, BaseAttackTime);\
SETATTR(data, double, AttackSpeed);\
SETATTR(data, double, Armor);\
SETATTR(data, double, Attack);\
SETATTR(data, double, AttackRange);\
SETATTR(data, double, SightRange);\
SETATTR(data, double, Bounty);\
SETATTR(data, double, bountyEXP);\
SETATTR(data, double, AtkPoint);\
SETATTR(data, double, AtkBackswing);\
SETATTR(data, double, ProjectileSpeed);\
SETATTR(data, AtkType, atktype)

typedef std::unordered_map<std::string, std::unordered_map<std::string, void*> > SpriteDataType;

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
        exp(0),_isDead(false),b_move(false), v_handle(NULL)
    {   
        _update_para();
    }

    Sprite() :LastAttackTime(-1),
        exp(0), gold(0), _isDead(false), b_move(false), canvas(NULL), v_handle(NULL) {}

    virtual ~Sprite(){
        remove_visual_ent();
        Py_XDECREF(v_handle);
    }

    inline void _update_para() {
        double AttackPerSecond = AttackSpeed * 0.01 / BaseAttackTime;
        AttackTime = 1 / AttackPerSecond;
    }

    virtual void step() = 0;
    virtual void draw() = 0;

    inline pos_tup pos_in_wnd() {
        return pos_tup(std::get<0>(location) * Config::game2window_scale * 0.5 + Config::windows_size * 0.5,
            std::get<1>(location) * Config::game2window_scale * 0.5 + Config::windows_size * 0.5);
    }

    void attack(Sprite* target);
    bool isAttacking();
    inline void set_move(pos_tup target) {
        b_move = true;
        move_target = target;
    }
    void move();
    bool damadged(Sprite* attacker, double dmg);
    void dead(Sprite*  attacker);
    void remove_visual_ent();

    static double S2Sdistance(const Sprite& s1,const Sprite& s2);

    inline cppSimulatorImp* get_engine() { return Engine; }

    inline double get_HP() { return HP; }
    inline double get_AttackTime() { return AttackTime; }
    inline double get_Attack() { return Attack; }
    inline Side get_side() { return side; }
    inline double get_SightRange() { return SightRange; }
    inline pos_tup get_location() { return location; }
    inline bool isDead(){return _isDead;}
    inline double get_ProjectileSpeed() { return ProjectileSpeed; }
    double TimeToDamage(const Sprite* s);

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
    double AtkPoint;
    double AtkBackswing;
    double ProjectileSpeed;
    double exp;
    double gold;
    bool _isDead;
    bool b_move;
    AtkType atktype;
    PyObject* v_handle;
    pos_tup move_target;
};

#endif//__SPRITE_H__