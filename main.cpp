#include <bits/stdc++.h>
#include <iostream>
#include <string>
#include <stdio.h>
#include <map>
#include "cl.hpp"

using namespace std;
#define int int64_t
#define FOR(i, n) for(int (i) = 0; (i) < (n); (i)++)
#define F0R(i, s, n) for(int (i) = (s); (i) < (n); (i)++)
#define nl "\n"
#define int128 __uint128_t

/**
 * Returns device from specific platform.
 */
Device OpenclHelper::get_device(uint platformIndex, uint deviceIndex) {
    VECTOR_CLASS<Platform> platforms;
    Platform::get(&platforms);
    if (platformIndex + 1 > platforms.size()) {
        printf("Invalid platform index: %d\n", platformIndex);
        return NULL;
    }

    VECTOR_CLASS<Device> devices;
    platforms[platformIndex].getDevices(CL_DEVICE_TYPE_ALL, &devices);
    if (deviceIndex + 1 > devices.size()) {
        printf("Invalid device index: %d\n", deviceIndex);
        return NULL;
    }

    return devices[deviceIndex];
}

signed main() {
    printf("hello World!\n");
}