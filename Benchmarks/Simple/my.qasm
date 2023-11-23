OPENQASM 2.0;
include "qelib1.inc";
qreg qr[2];
creg cr[2];
x qr[0];
cx qr[0],qr[1];
cx qr[1],qr[0];
cx qr[0],qr[1];
measure qr[0] -> cr[0];
measure qr[1] -> cr[1];


