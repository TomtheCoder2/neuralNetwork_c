struct Bar;

struct Foo {
    int tag;
    Bar *bp;
};

void setup() {
    Foo *hFoo = new Foo[10];

    Foo *dFoo;
    cudaMalloc(dFoo, sizeof(Foo) * 10);

    for (size_t i = 0; i != 10; ++i) {
        Bar *dBar;
        cudaMalloc(&dbar, sizeof(Bar));

        Bar b;  // automatic temporary -- we never keep a host copy of this
        cudaMemcpy(dBar, &b, sizeof(Bar));

        hFoo[i].bp = dBar;    // this is already a device pointer!
    }

    cudaMemcpy(dFoo, hFoo, sizeof(Foo) * 10);
}