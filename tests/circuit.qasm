// circuit_small.qasm
OPENQASM 2.0;
include "qelib1.inc";

// teach the parser what rzz is
gate rzz(theta) a,b {
    cx a,b;
    rz(theta) b;
    cx a,b;
}

qreg q[6];
creg c[6];

// layer 1
ry(0.671272670484062) q[0];
ry(0.383393090311768) q[1];
ry(0.390761880931935) q[2];
// entangle first three
cx q[0],q[1];
cx q[1],q[2];

// layer 2
ry(0.105246328897056) q[3];
cx q[2],q[3];

// layer 3
ry(0.738872295810560) q[4];
cx q[3],q[4];

// layer 4
ry(0.277312276421170) q[5];
cx q[4],q[5];

// now use rzz
rzz(1.5707963267948966) q[0],q[5];

// fan-in back to q0
cx q[5],q[4];
cx q[4],q[3];
cx q[3],q[2];
cx q[2],q[1];
cx q[1],q[0];

// final mixing
ry(0.55) q[0];

// seal it
barrier q;
