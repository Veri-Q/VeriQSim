OPENQASM 2.0;
include "qelib1.inc";
qreg qr[2];
creg cr[2];
cx qr[0], qr[1];
