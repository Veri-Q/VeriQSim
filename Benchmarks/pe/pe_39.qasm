OPENQASM 2.0;
include "qelib1.inc";
qreg q[40];
creg c[40];
h q[0];
h q[1];
h q[2];
h q[3];
h q[4];
h q[5];
h q[6];
h q[7];
h q[8];
h q[9];
h q[10];
h q[11];
h q[12];
h q[13];
h q[14];
h q[15];
h q[16];
h q[17];
h q[18];
h q[19];
h q[20];
h q[21];
h q[22];
h q[23];
h q[24];
h q[25];
h q[26];
h q[27];
h q[28];
h q[29];
h q[30];
h q[31];
h q[32];
h q[33];
h q[34];
h q[35];
h q[36];
h q[37];
h q[38];
cu1(pi/512) q[38],q[39];
cu1(pi/256) q[37],q[39];
cu1(pi/128) q[36],q[39];
cu1(pi/64) q[35],q[39];
cu1(pi/32) q[34],q[39];
cu1(pi/16) q[33],q[39];
cu1(pi/8) q[32],q[39];
cu1(pi/4) q[31],q[39];
cu1(pi/2) q[30],q[39];
cu1(pi) q[29],q[39];
cu1(2*pi) q[28],q[39];
cu1(4*pi) q[27],q[39];
cu1(8*pi) q[26],q[39];
cu1(16*pi) q[25],q[39];
cu1(32*pi) q[24],q[39];
cu1(64*pi) q[23],q[39];
cu1(128*pi) q[22],q[39];
cu1(256*pi) q[21],q[39];
cu1(512*pi) q[20],q[39];
cu1(1024*pi) q[19],q[39];
cu1(2048*pi) q[18],q[39];
cu1(4096*pi) q[17],q[39];
cu1(8192*pi) q[16],q[39];
cu1(16384*pi) q[15],q[39];
cu1(32768*pi) q[14],q[39];
cu1(65536*pi) q[13],q[39];
cu1(131072*pi) q[12],q[39];
cu1(262144*pi) q[11],q[39];
cu1(524288*pi) q[10],q[39];
cu1(1048576*pi) q[9],q[39];
cu1(2097152*pi) q[8],q[39];
cu1(4194304*pi) q[7],q[39];
cu1(8388608*pi) q[6],q[39];
cu1(16777216*pi) q[5],q[39];
cu1(33554432*pi) q[4],q[39];
cu1(67108864*pi) q[3],q[39];
cu1(134217728*pi) q[2],q[39];
cu1(268435456*pi) q[1],q[39];
cu1(536870912*pi) q[0],q[39];
h q[0];
cu1(-pi/2) q[0],q[1];
cu1(-pi/4) q[0],q[2];
cu1(-pi/8) q[0],q[3];
cu1(-pi/16) q[0],q[4];
cu1(-pi/32) q[0],q[5];
cu1(-pi/64) q[0],q[6];
cu1(-pi/128) q[0],q[7];
cu1(-pi/256) q[0],q[8];
cu1(-pi/512) q[0],q[9];
cu1(-pi/1024) q[0],q[10];
cu1(-pi/2048) q[0],q[11];
cu1(-pi/4096) q[0],q[12];
cu1(-pi/8192) q[0],q[13];
cu1(-pi/16384) q[0],q[14];
cu1(-pi/32768) q[0],q[15];
cu1(-pi/65536) q[0],q[16];
cu1(-pi/131072) q[0],q[17];
cu1(-pi/262144) q[0],q[18];
cu1(-pi/524288) q[0],q[19];
cu1(-pi/1048576) q[0],q[20];
cu1(-pi/2097152) q[0],q[21];
cu1(-pi/4194304) q[0],q[22];
cu1(-pi/8388608) q[0],q[23];
cu1(-pi/16777216) q[0],q[24];
cu1(-pi/33554432) q[0],q[25];
cu1(-pi/67108864) q[0],q[26];
cu1(-pi/134217728) q[0],q[27];
cu1(-pi/268435456) q[0],q[28];
cu1(-pi/536870912) q[0],q[29];
cu1(-pi/1073741824) q[0],q[30];
cu1(-pi/2147483648) q[0],q[31];
cu1(-pi/4294967296) q[0],q[32];
cu1(-pi/8589934592) q[0],q[33];
cu1(-pi/17179869184) q[0],q[34];
cu1(-pi/34359738368) q[0],q[35];
cu1(-pi/68719476736) q[0],q[36];
cu1(-pi/137438953472) q[0],q[37];
cu1(-pi/274877906944) q[0],q[38];
h q[1];
cu1(-pi/2) q[1],q[2];
cu1(-pi/4) q[1],q[3];
cu1(-pi/8) q[1],q[4];
cu1(-pi/16) q[1],q[5];
cu1(-pi/32) q[1],q[6];
cu1(-pi/64) q[1],q[7];
cu1(-pi/128) q[1],q[8];
cu1(-pi/256) q[1],q[9];
cu1(-pi/512) q[1],q[10];
cu1(-pi/1024) q[1],q[11];
cu1(-pi/2048) q[1],q[12];
cu1(-pi/4096) q[1],q[13];
cu1(-pi/8192) q[1],q[14];
cu1(-pi/16384) q[1],q[15];
cu1(-pi/32768) q[1],q[16];
cu1(-pi/65536) q[1],q[17];
cu1(-pi/131072) q[1],q[18];
cu1(-pi/262144) q[1],q[19];
cu1(-pi/524288) q[1],q[20];
cu1(-pi/1048576) q[1],q[21];
cu1(-pi/2097152) q[1],q[22];
cu1(-pi/4194304) q[1],q[23];
cu1(-pi/8388608) q[1],q[24];
cu1(-pi/16777216) q[1],q[25];
cu1(-pi/33554432) q[1],q[26];
cu1(-pi/67108864) q[1],q[27];
cu1(-pi/134217728) q[1],q[28];
cu1(-pi/268435456) q[1],q[29];
cu1(-pi/536870912) q[1],q[30];
cu1(-pi/1073741824) q[1],q[31];
cu1(-pi/2147483648) q[1],q[32];
cu1(-pi/4294967296) q[1],q[33];
cu1(-pi/8589934592) q[1],q[34];
cu1(-pi/17179869184) q[1],q[35];
cu1(-pi/34359738368) q[1],q[36];
cu1(-pi/68719476736) q[1],q[37];
cu1(-pi/137438953472) q[1],q[38];
h q[2];
cu1(-pi/2) q[2],q[3];
cu1(-pi/4) q[2],q[4];
cu1(-pi/8) q[2],q[5];
cu1(-pi/16) q[2],q[6];
cu1(-pi/32) q[2],q[7];
cu1(-pi/64) q[2],q[8];
cu1(-pi/128) q[2],q[9];
cu1(-pi/256) q[2],q[10];
cu1(-pi/512) q[2],q[11];
cu1(-pi/1024) q[2],q[12];
cu1(-pi/2048) q[2],q[13];
cu1(-pi/4096) q[2],q[14];
cu1(-pi/8192) q[2],q[15];
cu1(-pi/16384) q[2],q[16];
cu1(-pi/32768) q[2],q[17];
cu1(-pi/65536) q[2],q[18];
cu1(-pi/131072) q[2],q[19];
cu1(-pi/262144) q[2],q[20];
cu1(-pi/524288) q[2],q[21];
cu1(-pi/1048576) q[2],q[22];
cu1(-pi/2097152) q[2],q[23];
cu1(-pi/4194304) q[2],q[24];
cu1(-pi/8388608) q[2],q[25];
cu1(-pi/16777216) q[2],q[26];
cu1(-pi/33554432) q[2],q[27];
cu1(-pi/67108864) q[2],q[28];
cu1(-pi/134217728) q[2],q[29];
cu1(-pi/268435456) q[2],q[30];
cu1(-pi/536870912) q[2],q[31];
cu1(-pi/1073741824) q[2],q[32];
cu1(-pi/2147483648) q[2],q[33];
cu1(-pi/4294967296) q[2],q[34];
cu1(-pi/8589934592) q[2],q[35];
cu1(-pi/17179869184) q[2],q[36];
cu1(-pi/34359738368) q[2],q[37];
cu1(-pi/68719476736) q[2],q[38];
h q[3];
cu1(-pi/2) q[3],q[4];
cu1(-pi/4) q[3],q[5];
cu1(-pi/8) q[3],q[6];
cu1(-pi/16) q[3],q[7];
cu1(-pi/32) q[3],q[8];
cu1(-pi/64) q[3],q[9];
cu1(-pi/128) q[3],q[10];
cu1(-pi/256) q[3],q[11];
cu1(-pi/512) q[3],q[12];
cu1(-pi/1024) q[3],q[13];
cu1(-pi/2048) q[3],q[14];
cu1(-pi/4096) q[3],q[15];
cu1(-pi/8192) q[3],q[16];
cu1(-pi/16384) q[3],q[17];
cu1(-pi/32768) q[3],q[18];
cu1(-pi/65536) q[3],q[19];
cu1(-pi/131072) q[3],q[20];
cu1(-pi/262144) q[3],q[21];
cu1(-pi/524288) q[3],q[22];
cu1(-pi/1048576) q[3],q[23];
cu1(-pi/2097152) q[3],q[24];
cu1(-pi/4194304) q[3],q[25];
cu1(-pi/8388608) q[3],q[26];
cu1(-pi/16777216) q[3],q[27];
cu1(-pi/33554432) q[3],q[28];
cu1(-pi/67108864) q[3],q[29];
cu1(-pi/134217728) q[3],q[30];
cu1(-pi/268435456) q[3],q[31];
cu1(-pi/536870912) q[3],q[32];
cu1(-pi/1073741824) q[3],q[33];
cu1(-pi/2147483648) q[3],q[34];
cu1(-pi/4294967296) q[3],q[35];
cu1(-pi/8589934592) q[3],q[36];
cu1(-pi/17179869184) q[3],q[37];
cu1(-pi/34359738368) q[3],q[38];
h q[4];
cu1(-pi/2) q[4],q[5];
cu1(-pi/4) q[4],q[6];
cu1(-pi/8) q[4],q[7];
cu1(-pi/16) q[4],q[8];
cu1(-pi/32) q[4],q[9];
cu1(-pi/64) q[4],q[10];
cu1(-pi/128) q[4],q[11];
cu1(-pi/256) q[4],q[12];
cu1(-pi/512) q[4],q[13];
cu1(-pi/1024) q[4],q[14];
cu1(-pi/2048) q[4],q[15];
cu1(-pi/4096) q[4],q[16];
cu1(-pi/8192) q[4],q[17];
cu1(-pi/16384) q[4],q[18];
cu1(-pi/32768) q[4],q[19];
cu1(-pi/65536) q[4],q[20];
cu1(-pi/131072) q[4],q[21];
cu1(-pi/262144) q[4],q[22];
cu1(-pi/524288) q[4],q[23];
cu1(-pi/1048576) q[4],q[24];
cu1(-pi/2097152) q[4],q[25];
cu1(-pi/4194304) q[4],q[26];
cu1(-pi/8388608) q[4],q[27];
cu1(-pi/16777216) q[4],q[28];
cu1(-pi/33554432) q[4],q[29];
cu1(-pi/67108864) q[4],q[30];
cu1(-pi/134217728) q[4],q[31];
cu1(-pi/268435456) q[4],q[32];
cu1(-pi/536870912) q[4],q[33];
cu1(-pi/1073741824) q[4],q[34];
cu1(-pi/2147483648) q[4],q[35];
cu1(-pi/4294967296) q[4],q[36];
cu1(-pi/8589934592) q[4],q[37];
cu1(-pi/17179869184) q[4],q[38];
h q[5];
cu1(-pi/2) q[5],q[6];
cu1(-pi/4) q[5],q[7];
cu1(-pi/8) q[5],q[8];
cu1(-pi/16) q[5],q[9];
cu1(-pi/32) q[5],q[10];
cu1(-pi/64) q[5],q[11];
cu1(-pi/128) q[5],q[12];
cu1(-pi/256) q[5],q[13];
cu1(-pi/512) q[5],q[14];
cu1(-pi/1024) q[5],q[15];
cu1(-pi/2048) q[5],q[16];
cu1(-pi/4096) q[5],q[17];
cu1(-pi/8192) q[5],q[18];
cu1(-pi/16384) q[5],q[19];
cu1(-pi/32768) q[5],q[20];
cu1(-pi/65536) q[5],q[21];
cu1(-pi/131072) q[5],q[22];
cu1(-pi/262144) q[5],q[23];
cu1(-pi/524288) q[5],q[24];
cu1(-pi/1048576) q[5],q[25];
cu1(-pi/2097152) q[5],q[26];
cu1(-pi/4194304) q[5],q[27];
cu1(-pi/8388608) q[5],q[28];
cu1(-pi/16777216) q[5],q[29];
cu1(-pi/33554432) q[5],q[30];
cu1(-pi/67108864) q[5],q[31];
cu1(-pi/134217728) q[5],q[32];
cu1(-pi/268435456) q[5],q[33];
cu1(-pi/536870912) q[5],q[34];
cu1(-pi/1073741824) q[5],q[35];
cu1(-pi/2147483648) q[5],q[36];
cu1(-pi/4294967296) q[5],q[37];
cu1(-pi/8589934592) q[5],q[38];
h q[6];
cu1(-pi/2) q[6],q[7];
cu1(-pi/4) q[6],q[8];
cu1(-pi/8) q[6],q[9];
cu1(-pi/16) q[6],q[10];
cu1(-pi/32) q[6],q[11];
cu1(-pi/64) q[6],q[12];
cu1(-pi/128) q[6],q[13];
cu1(-pi/256) q[6],q[14];
cu1(-pi/512) q[6],q[15];
cu1(-pi/1024) q[6],q[16];
cu1(-pi/2048) q[6],q[17];
cu1(-pi/4096) q[6],q[18];
cu1(-pi/8192) q[6],q[19];
cu1(-pi/16384) q[6],q[20];
cu1(-pi/32768) q[6],q[21];
cu1(-pi/65536) q[6],q[22];
cu1(-pi/131072) q[6],q[23];
cu1(-pi/262144) q[6],q[24];
cu1(-pi/524288) q[6],q[25];
cu1(-pi/1048576) q[6],q[26];
cu1(-pi/2097152) q[6],q[27];
cu1(-pi/4194304) q[6],q[28];
cu1(-pi/8388608) q[6],q[29];
cu1(-pi/16777216) q[6],q[30];
cu1(-pi/33554432) q[6],q[31];
cu1(-pi/67108864) q[6],q[32];
cu1(-pi/134217728) q[6],q[33];
cu1(-pi/268435456) q[6],q[34];
cu1(-pi/536870912) q[6],q[35];
cu1(-pi/1073741824) q[6],q[36];
cu1(-pi/2147483648) q[6],q[37];
cu1(-pi/4294967296) q[6],q[38];
h q[7];
cu1(-pi/2) q[7],q[8];
cu1(-pi/4) q[7],q[9];
cu1(-pi/8) q[7],q[10];
cu1(-pi/16) q[7],q[11];
cu1(-pi/32) q[7],q[12];
cu1(-pi/64) q[7],q[13];
cu1(-pi/128) q[7],q[14];
cu1(-pi/256) q[7],q[15];
cu1(-pi/512) q[7],q[16];
cu1(-pi/1024) q[7],q[17];
cu1(-pi/2048) q[7],q[18];
cu1(-pi/4096) q[7],q[19];
cu1(-pi/8192) q[7],q[20];
cu1(-pi/16384) q[7],q[21];
cu1(-pi/32768) q[7],q[22];
cu1(-pi/65536) q[7],q[23];
cu1(-pi/131072) q[7],q[24];
cu1(-pi/262144) q[7],q[25];
cu1(-pi/524288) q[7],q[26];
cu1(-pi/1048576) q[7],q[27];
cu1(-pi/2097152) q[7],q[28];
cu1(-pi/4194304) q[7],q[29];
cu1(-pi/8388608) q[7],q[30];
cu1(-pi/16777216) q[7],q[31];
cu1(-pi/33554432) q[7],q[32];
cu1(-pi/67108864) q[7],q[33];
cu1(-pi/134217728) q[7],q[34];
cu1(-pi/268435456) q[7],q[35];
cu1(-pi/536870912) q[7],q[36];
cu1(-pi/1073741824) q[7],q[37];
cu1(-pi/2147483648) q[7],q[38];
h q[8];
cu1(-pi/2) q[8],q[9];
cu1(-pi/4) q[8],q[10];
cu1(-pi/8) q[8],q[11];
cu1(-pi/16) q[8],q[12];
cu1(-pi/32) q[8],q[13];
cu1(-pi/64) q[8],q[14];
cu1(-pi/128) q[8],q[15];
cu1(-pi/256) q[8],q[16];
cu1(-pi/512) q[8],q[17];
cu1(-pi/1024) q[8],q[18];
cu1(-pi/2048) q[8],q[19];
cu1(-pi/4096) q[8],q[20];
cu1(-pi/8192) q[8],q[21];
cu1(-pi/16384) q[8],q[22];
cu1(-pi/32768) q[8],q[23];
cu1(-pi/65536) q[8],q[24];
cu1(-pi/131072) q[8],q[25];
cu1(-pi/262144) q[8],q[26];
cu1(-pi/524288) q[8],q[27];
cu1(-pi/1048576) q[8],q[28];
cu1(-pi/2097152) q[8],q[29];
cu1(-pi/4194304) q[8],q[30];
cu1(-pi/8388608) q[8],q[31];
cu1(-pi/16777216) q[8],q[32];
cu1(-pi/33554432) q[8],q[33];
cu1(-pi/67108864) q[8],q[34];
cu1(-pi/134217728) q[8],q[35];
cu1(-pi/268435456) q[8],q[36];
cu1(-pi/536870912) q[8],q[37];
cu1(-pi/1073741824) q[8],q[38];
h q[9];
cu1(-pi/2) q[9],q[10];
cu1(-pi/4) q[9],q[11];
cu1(-pi/8) q[9],q[12];
cu1(-pi/16) q[9],q[13];
cu1(-pi/32) q[9],q[14];
cu1(-pi/64) q[9],q[15];
cu1(-pi/128) q[9],q[16];
cu1(-pi/256) q[9],q[17];
cu1(-pi/512) q[9],q[18];
cu1(-pi/1024) q[9],q[19];
cu1(-pi/2048) q[9],q[20];
cu1(-pi/4096) q[9],q[21];
cu1(-pi/8192) q[9],q[22];
cu1(-pi/16384) q[9],q[23];
cu1(-pi/32768) q[9],q[24];
cu1(-pi/65536) q[9],q[25];
cu1(-pi/131072) q[9],q[26];
cu1(-pi/262144) q[9],q[27];
cu1(-pi/524288) q[9],q[28];
cu1(-pi/1048576) q[9],q[29];
cu1(-pi/2097152) q[9],q[30];
cu1(-pi/4194304) q[9],q[31];
cu1(-pi/8388608) q[9],q[32];
cu1(-pi/16777216) q[9],q[33];
cu1(-pi/33554432) q[9],q[34];
cu1(-pi/67108864) q[9],q[35];
cu1(-pi/134217728) q[9],q[36];
cu1(-pi/268435456) q[9],q[37];
cu1(-pi/536870912) q[9],q[38];
h q[10];
cu1(-pi/2) q[10],q[11];
cu1(-pi/4) q[10],q[12];
cu1(-pi/8) q[10],q[13];
cu1(-pi/16) q[10],q[14];
cu1(-pi/32) q[10],q[15];
cu1(-pi/64) q[10],q[16];
cu1(-pi/128) q[10],q[17];
cu1(-pi/256) q[10],q[18];
cu1(-pi/512) q[10],q[19];
cu1(-pi/1024) q[10],q[20];
cu1(-pi/2048) q[10],q[21];
cu1(-pi/4096) q[10],q[22];
cu1(-pi/8192) q[10],q[23];
cu1(-pi/16384) q[10],q[24];
cu1(-pi/32768) q[10],q[25];
cu1(-pi/65536) q[10],q[26];
cu1(-pi/131072) q[10],q[27];
cu1(-pi/262144) q[10],q[28];
cu1(-pi/524288) q[10],q[29];
cu1(-pi/1048576) q[10],q[30];
cu1(-pi/2097152) q[10],q[31];
cu1(-pi/4194304) q[10],q[32];
cu1(-pi/8388608) q[10],q[33];
cu1(-pi/16777216) q[10],q[34];
cu1(-pi/33554432) q[10],q[35];
cu1(-pi/67108864) q[10],q[36];
cu1(-pi/134217728) q[10],q[37];
cu1(-pi/268435456) q[10],q[38];
h q[11];
cu1(-pi/2) q[11],q[12];
cu1(-pi/4) q[11],q[13];
cu1(-pi/8) q[11],q[14];
cu1(-pi/16) q[11],q[15];
cu1(-pi/32) q[11],q[16];
cu1(-pi/64) q[11],q[17];
cu1(-pi/128) q[11],q[18];
cu1(-pi/256) q[11],q[19];
cu1(-pi/512) q[11],q[20];
cu1(-pi/1024) q[11],q[21];
cu1(-pi/2048) q[11],q[22];
cu1(-pi/4096) q[11],q[23];
cu1(-pi/8192) q[11],q[24];
cu1(-pi/16384) q[11],q[25];
cu1(-pi/32768) q[11],q[26];
cu1(-pi/65536) q[11],q[27];
cu1(-pi/131072) q[11],q[28];
cu1(-pi/262144) q[11],q[29];
cu1(-pi/524288) q[11],q[30];
cu1(-pi/1048576) q[11],q[31];
cu1(-pi/2097152) q[11],q[32];
cu1(-pi/4194304) q[11],q[33];
cu1(-pi/8388608) q[11],q[34];
cu1(-pi/16777216) q[11],q[35];
cu1(-pi/33554432) q[11],q[36];
cu1(-pi/67108864) q[11],q[37];
cu1(-pi/134217728) q[11],q[38];
h q[12];
cu1(-pi/2) q[12],q[13];
cu1(-pi/4) q[12],q[14];
cu1(-pi/8) q[12],q[15];
cu1(-pi/16) q[12],q[16];
cu1(-pi/32) q[12],q[17];
cu1(-pi/64) q[12],q[18];
cu1(-pi/128) q[12],q[19];
cu1(-pi/256) q[12],q[20];
cu1(-pi/512) q[12],q[21];
cu1(-pi/1024) q[12],q[22];
cu1(-pi/2048) q[12],q[23];
cu1(-pi/4096) q[12],q[24];
cu1(-pi/8192) q[12],q[25];
cu1(-pi/16384) q[12],q[26];
cu1(-pi/32768) q[12],q[27];
cu1(-pi/65536) q[12],q[28];
cu1(-pi/131072) q[12],q[29];
cu1(-pi/262144) q[12],q[30];
cu1(-pi/524288) q[12],q[31];
cu1(-pi/1048576) q[12],q[32];
cu1(-pi/2097152) q[12],q[33];
cu1(-pi/4194304) q[12],q[34];
cu1(-pi/8388608) q[12],q[35];
cu1(-pi/16777216) q[12],q[36];
cu1(-pi/33554432) q[12],q[37];
cu1(-pi/67108864) q[12],q[38];
h q[13];
cu1(-pi/2) q[13],q[14];
cu1(-pi/4) q[13],q[15];
cu1(-pi/8) q[13],q[16];
cu1(-pi/16) q[13],q[17];
cu1(-pi/32) q[13],q[18];
cu1(-pi/64) q[13],q[19];
cu1(-pi/128) q[13],q[20];
cu1(-pi/256) q[13],q[21];
cu1(-pi/512) q[13],q[22];
cu1(-pi/1024) q[13],q[23];
cu1(-pi/2048) q[13],q[24];
cu1(-pi/4096) q[13],q[25];
cu1(-pi/8192) q[13],q[26];
cu1(-pi/16384) q[13],q[27];
cu1(-pi/32768) q[13],q[28];
cu1(-pi/65536) q[13],q[29];
cu1(-pi/131072) q[13],q[30];
cu1(-pi/262144) q[13],q[31];
cu1(-pi/524288) q[13],q[32];
cu1(-pi/1048576) q[13],q[33];
cu1(-pi/2097152) q[13],q[34];
cu1(-pi/4194304) q[13],q[35];
cu1(-pi/8388608) q[13],q[36];
cu1(-pi/16777216) q[13],q[37];
cu1(-pi/33554432) q[13],q[38];
h q[14];
cu1(-pi/2) q[14],q[15];
cu1(-pi/4) q[14],q[16];
cu1(-pi/8) q[14],q[17];
cu1(-pi/16) q[14],q[18];
cu1(-pi/32) q[14],q[19];
cu1(-pi/64) q[14],q[20];
cu1(-pi/128) q[14],q[21];
cu1(-pi/256) q[14],q[22];
cu1(-pi/512) q[14],q[23];
cu1(-pi/1024) q[14],q[24];
cu1(-pi/2048) q[14],q[25];
cu1(-pi/4096) q[14],q[26];
cu1(-pi/8192) q[14],q[27];
cu1(-pi/16384) q[14],q[28];
cu1(-pi/32768) q[14],q[29];
cu1(-pi/65536) q[14],q[30];
cu1(-pi/131072) q[14],q[31];
cu1(-pi/262144) q[14],q[32];
cu1(-pi/524288) q[14],q[33];
cu1(-pi/1048576) q[14],q[34];
cu1(-pi/2097152) q[14],q[35];
cu1(-pi/4194304) q[14],q[36];
cu1(-pi/8388608) q[14],q[37];
cu1(-pi/16777216) q[14],q[38];
h q[15];
cu1(-pi/2) q[15],q[16];
cu1(-pi/4) q[15],q[17];
cu1(-pi/8) q[15],q[18];
cu1(-pi/16) q[15],q[19];
cu1(-pi/32) q[15],q[20];
cu1(-pi/64) q[15],q[21];
cu1(-pi/128) q[15],q[22];
cu1(-pi/256) q[15],q[23];
cu1(-pi/512) q[15],q[24];
cu1(-pi/1024) q[15],q[25];
cu1(-pi/2048) q[15],q[26];
cu1(-pi/4096) q[15],q[27];
cu1(-pi/8192) q[15],q[28];
cu1(-pi/16384) q[15],q[29];
cu1(-pi/32768) q[15],q[30];
cu1(-pi/65536) q[15],q[31];
cu1(-pi/131072) q[15],q[32];
cu1(-pi/262144) q[15],q[33];
cu1(-pi/524288) q[15],q[34];
cu1(-pi/1048576) q[15],q[35];
cu1(-pi/2097152) q[15],q[36];
cu1(-pi/4194304) q[15],q[37];
cu1(-pi/8388608) q[15],q[38];
h q[16];
cu1(-pi/2) q[16],q[17];
cu1(-pi/4) q[16],q[18];
cu1(-pi/8) q[16],q[19];
cu1(-pi/16) q[16],q[20];
cu1(-pi/32) q[16],q[21];
cu1(-pi/64) q[16],q[22];
cu1(-pi/128) q[16],q[23];
cu1(-pi/256) q[16],q[24];
cu1(-pi/512) q[16],q[25];
cu1(-pi/1024) q[16],q[26];
cu1(-pi/2048) q[16],q[27];
cu1(-pi/4096) q[16],q[28];
cu1(-pi/8192) q[16],q[29];
cu1(-pi/16384) q[16],q[30];
cu1(-pi/32768) q[16],q[31];
cu1(-pi/65536) q[16],q[32];
cu1(-pi/131072) q[16],q[33];
cu1(-pi/262144) q[16],q[34];
cu1(-pi/524288) q[16],q[35];
cu1(-pi/1048576) q[16],q[36];
cu1(-pi/2097152) q[16],q[37];
cu1(-pi/4194304) q[16],q[38];
h q[17];
cu1(-pi/2) q[17],q[18];
cu1(-pi/4) q[17],q[19];
cu1(-pi/8) q[17],q[20];
cu1(-pi/16) q[17],q[21];
cu1(-pi/32) q[17],q[22];
cu1(-pi/64) q[17],q[23];
cu1(-pi/128) q[17],q[24];
cu1(-pi/256) q[17],q[25];
cu1(-pi/512) q[17],q[26];
cu1(-pi/1024) q[17],q[27];
cu1(-pi/2048) q[17],q[28];
cu1(-pi/4096) q[17],q[29];
cu1(-pi/8192) q[17],q[30];
cu1(-pi/16384) q[17],q[31];
cu1(-pi/32768) q[17],q[32];
cu1(-pi/65536) q[17],q[33];
cu1(-pi/131072) q[17],q[34];
cu1(-pi/262144) q[17],q[35];
cu1(-pi/524288) q[17],q[36];
cu1(-pi/1048576) q[17],q[37];
cu1(-pi/2097152) q[17],q[38];
h q[18];
cu1(-pi/2) q[18],q[19];
cu1(-pi/4) q[18],q[20];
cu1(-pi/8) q[18],q[21];
cu1(-pi/16) q[18],q[22];
cu1(-pi/32) q[18],q[23];
cu1(-pi/64) q[18],q[24];
cu1(-pi/128) q[18],q[25];
cu1(-pi/256) q[18],q[26];
cu1(-pi/512) q[18],q[27];
cu1(-pi/1024) q[18],q[28];
cu1(-pi/2048) q[18],q[29];
cu1(-pi/4096) q[18],q[30];
cu1(-pi/8192) q[18],q[31];
cu1(-pi/16384) q[18],q[32];
cu1(-pi/32768) q[18],q[33];
cu1(-pi/65536) q[18],q[34];
cu1(-pi/131072) q[18],q[35];
cu1(-pi/262144) q[18],q[36];
cu1(-pi/524288) q[18],q[37];
cu1(-pi/1048576) q[18],q[38];
h q[19];
cu1(-pi/2) q[19],q[20];
cu1(-pi/4) q[19],q[21];
cu1(-pi/8) q[19],q[22];
cu1(-pi/16) q[19],q[23];
cu1(-pi/32) q[19],q[24];
cu1(-pi/64) q[19],q[25];
cu1(-pi/128) q[19],q[26];
cu1(-pi/256) q[19],q[27];
cu1(-pi/512) q[19],q[28];
cu1(-pi/1024) q[19],q[29];
cu1(-pi/2048) q[19],q[30];
cu1(-pi/4096) q[19],q[31];
cu1(-pi/8192) q[19],q[32];
cu1(-pi/16384) q[19],q[33];
cu1(-pi/32768) q[19],q[34];
cu1(-pi/65536) q[19],q[35];
cu1(-pi/131072) q[19],q[36];
cu1(-pi/262144) q[19],q[37];
cu1(-pi/524288) q[19],q[38];
h q[20];
cu1(-pi/2) q[20],q[21];
cu1(-pi/4) q[20],q[22];
cu1(-pi/8) q[20],q[23];
cu1(-pi/16) q[20],q[24];
cu1(-pi/32) q[20],q[25];
cu1(-pi/64) q[20],q[26];
cu1(-pi/128) q[20],q[27];
cu1(-pi/256) q[20],q[28];
cu1(-pi/512) q[20],q[29];
cu1(-pi/1024) q[20],q[30];
cu1(-pi/2048) q[20],q[31];
cu1(-pi/4096) q[20],q[32];
cu1(-pi/8192) q[20],q[33];
cu1(-pi/16384) q[20],q[34];
cu1(-pi/32768) q[20],q[35];
cu1(-pi/65536) q[20],q[36];
cu1(-pi/131072) q[20],q[37];
cu1(-pi/262144) q[20],q[38];
h q[21];
cu1(-pi/2) q[21],q[22];
cu1(-pi/4) q[21],q[23];
cu1(-pi/8) q[21],q[24];
cu1(-pi/16) q[21],q[25];
cu1(-pi/32) q[21],q[26];
cu1(-pi/64) q[21],q[27];
cu1(-pi/128) q[21],q[28];
cu1(-pi/256) q[21],q[29];
cu1(-pi/512) q[21],q[30];
cu1(-pi/1024) q[21],q[31];
cu1(-pi/2048) q[21],q[32];
cu1(-pi/4096) q[21],q[33];
cu1(-pi/8192) q[21],q[34];
cu1(-pi/16384) q[21],q[35];
cu1(-pi/32768) q[21],q[36];
cu1(-pi/65536) q[21],q[37];
cu1(-pi/131072) q[21],q[38];
h q[22];
cu1(-pi/2) q[22],q[23];
cu1(-pi/4) q[22],q[24];
cu1(-pi/8) q[22],q[25];
cu1(-pi/16) q[22],q[26];
cu1(-pi/32) q[22],q[27];
cu1(-pi/64) q[22],q[28];
cu1(-pi/128) q[22],q[29];
cu1(-pi/256) q[22],q[30];
cu1(-pi/512) q[22],q[31];
cu1(-pi/1024) q[22],q[32];
cu1(-pi/2048) q[22],q[33];
cu1(-pi/4096) q[22],q[34];
cu1(-pi/8192) q[22],q[35];
cu1(-pi/16384) q[22],q[36];
cu1(-pi/32768) q[22],q[37];
cu1(-pi/65536) q[22],q[38];
h q[23];
cu1(-pi/2) q[23],q[24];
cu1(-pi/4) q[23],q[25];
cu1(-pi/8) q[23],q[26];
cu1(-pi/16) q[23],q[27];
cu1(-pi/32) q[23],q[28];
cu1(-pi/64) q[23],q[29];
cu1(-pi/128) q[23],q[30];
cu1(-pi/256) q[23],q[31];
cu1(-pi/512) q[23],q[32];
cu1(-pi/1024) q[23],q[33];
cu1(-pi/2048) q[23],q[34];
cu1(-pi/4096) q[23],q[35];
cu1(-pi/8192) q[23],q[36];
cu1(-pi/16384) q[23],q[37];
cu1(-pi/32768) q[23],q[38];
h q[24];
cu1(-pi/2) q[24],q[25];
cu1(-pi/4) q[24],q[26];
cu1(-pi/8) q[24],q[27];
cu1(-pi/16) q[24],q[28];
cu1(-pi/32) q[24],q[29];
cu1(-pi/64) q[24],q[30];
cu1(-pi/128) q[24],q[31];
cu1(-pi/256) q[24],q[32];
cu1(-pi/512) q[24],q[33];
cu1(-pi/1024) q[24],q[34];
cu1(-pi/2048) q[24],q[35];
cu1(-pi/4096) q[24],q[36];
cu1(-pi/8192) q[24],q[37];
cu1(-pi/16384) q[24],q[38];
h q[25];
cu1(-pi/2) q[25],q[26];
cu1(-pi/4) q[25],q[27];
cu1(-pi/8) q[25],q[28];
cu1(-pi/16) q[25],q[29];
cu1(-pi/32) q[25],q[30];
cu1(-pi/64) q[25],q[31];
cu1(-pi/128) q[25],q[32];
cu1(-pi/256) q[25],q[33];
cu1(-pi/512) q[25],q[34];
cu1(-pi/1024) q[25],q[35];
cu1(-pi/2048) q[25],q[36];
cu1(-pi/4096) q[25],q[37];
cu1(-pi/8192) q[25],q[38];
h q[26];
cu1(-pi/2) q[26],q[27];
cu1(-pi/4) q[26],q[28];
cu1(-pi/8) q[26],q[29];
cu1(-pi/16) q[26],q[30];
cu1(-pi/32) q[26],q[31];
cu1(-pi/64) q[26],q[32];
cu1(-pi/128) q[26],q[33];
cu1(-pi/256) q[26],q[34];
cu1(-pi/512) q[26],q[35];
cu1(-pi/1024) q[26],q[36];
cu1(-pi/2048) q[26],q[37];
cu1(-pi/4096) q[26],q[38];
h q[27];
cu1(-pi/2) q[27],q[28];
cu1(-pi/4) q[27],q[29];
cu1(-pi/8) q[27],q[30];
cu1(-pi/16) q[27],q[31];
cu1(-pi/32) q[27],q[32];
cu1(-pi/64) q[27],q[33];
cu1(-pi/128) q[27],q[34];
cu1(-pi/256) q[27],q[35];
cu1(-pi/512) q[27],q[36];
cu1(-pi/1024) q[27],q[37];
cu1(-pi/2048) q[27],q[38];
h q[28];
cu1(-pi/2) q[28],q[29];
cu1(-pi/4) q[28],q[30];
cu1(-pi/8) q[28],q[31];
cu1(-pi/16) q[28],q[32];
cu1(-pi/32) q[28],q[33];
cu1(-pi/64) q[28],q[34];
cu1(-pi/128) q[28],q[35];
cu1(-pi/256) q[28],q[36];
cu1(-pi/512) q[28],q[37];
cu1(-pi/1024) q[28],q[38];
h q[29];
cu1(-pi/2) q[29],q[30];
cu1(-pi/4) q[29],q[31];
cu1(-pi/8) q[29],q[32];
cu1(-pi/16) q[29],q[33];
cu1(-pi/32) q[29],q[34];
cu1(-pi/64) q[29],q[35];
cu1(-pi/128) q[29],q[36];
cu1(-pi/256) q[29],q[37];
cu1(-pi/512) q[29],q[38];
h q[30];
cu1(-pi/2) q[30],q[31];
cu1(-pi/4) q[30],q[32];
cu1(-pi/8) q[30],q[33];
cu1(-pi/16) q[30],q[34];
cu1(-pi/32) q[30],q[35];
cu1(-pi/64) q[30],q[36];
cu1(-pi/128) q[30],q[37];
cu1(-pi/256) q[30],q[38];
h q[31];
cu1(-pi/2) q[31],q[32];
cu1(-pi/4) q[31],q[33];
cu1(-pi/8) q[31],q[34];
cu1(-pi/16) q[31],q[35];
cu1(-pi/32) q[31],q[36];
cu1(-pi/64) q[31],q[37];
cu1(-pi/128) q[31],q[38];
h q[32];
cu1(-pi/2) q[32],q[33];
cu1(-pi/4) q[32],q[34];
cu1(-pi/8) q[32],q[35];
cu1(-pi/16) q[32],q[36];
cu1(-pi/32) q[32],q[37];
cu1(-pi/64) q[32],q[38];
h q[33];
cu1(-pi/2) q[33],q[34];
cu1(-pi/4) q[33],q[35];
cu1(-pi/8) q[33],q[36];
cu1(-pi/16) q[33],q[37];
cu1(-pi/32) q[33],q[38];
h q[34];
cu1(-pi/2) q[34],q[35];
cu1(-pi/4) q[34],q[36];
cu1(-pi/8) q[34],q[37];
cu1(-pi/16) q[34],q[38];
h q[35];
cu1(-pi/2) q[35],q[36];
cu1(-pi/4) q[35],q[37];
cu1(-pi/8) q[35],q[38];
h q[36];
cu1(-pi/2) q[36],q[37];
cu1(-pi/4) q[36],q[38];
h q[37];
cu1(-pi/2) q[37],q[38];
h q[38];
