template<class T>
struct Fenwick {
    int n;
    vector<ll> tr;
    Fenwick(int sz) : n(sz), tr(sz + 1) { }
    void add(int x, const T& v) {
        while(x <= n) tr[x] += v, x += x & -x;
    }
    T query(int l, int r) {
        assert(l <= r);
        T res{};
        for(int x = r; x; x -= x & -x) res += tr[x];
        for(int x = l - 1; x; x -= x & -x) res -= tr[x];
        return res;
    }
    int lower_bound(T k) {
        int x = 0;
        for(int i = 1 << __lg(n); i; i >>= 1)
            if(x + i <= n && tr[x + i] < k)
                k -= tr[x += i];
        return x + 1;
    }
};
