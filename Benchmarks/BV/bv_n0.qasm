//Bernstein-Vazirani with 2 qubits.
//Hidden string is 1
OPENQASM 2.0;
include "qelib1.inc";
qreg qr[1];
creg cr[1];
x qr[0];
measure qr[0] -> cr[0];
