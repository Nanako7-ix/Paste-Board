///My_code
#define _My_code cin.tie(0), cout.tie(0)->sync_with_stdio(0);
#include <bits/stdc++.h>
using namespace std;
#ifndef ONLINE_JUDGE
#include"Others/M.h"
#endif

int main() { _My_code
}
///n1
n + 1
///pii
pair<int, int>
///vi
vector<int>
///vector<int>
vector<int>
///vector<long long>
vector<long long>
///ll
long long
///int
int
///inf
1 << 29
///kmp
struct kmp {
    vector<int> nxt, ans;
    kmp(string s, string p) {
        nxt = vector<int>(s.size()), ans = vector<int>();
        for (int i = 1, j = 0; i < p.size(); p[i] == p[j] && ++j, nxt[i++] = j)
            while (j && p[i] != p[j]) j = nxt[j - 1];
        for (int i = 0, j = 0; i < s.size(); s[i++] == p[j] && ++j == p.size() && (ans.push_back(i - j), j = nxt[j - 1]))
            while (j && s[i] != p[j]) j = nxt[j - 1];
    }
};

///edge
int n, m; cin >> n >> m;
vector<vector<int>> edge(n + 1);
for (int i = 0; i < m; ++i)
    if (int j, k; cin >> j >> k, j != k)
        edge[j].push_back(k), edge[k].push_back(j);

///edgew
int n, m; cin >> n >> m;
vector<vector<pair<int, int>>> edge(n + 1);
for (int i = 0; i < m; ++i)
    if (int j, k, l; cin >> j >> k >> l, j != k)
        edge[j].push_back({k, l}), edge[k].push_back({j, l});

///dijstra 单源最短路 O(nlogm)
vector<int> last(n + 1, 0);
vector<int> ans(n + 1, 1 << 29), vis(n + 1);
priority_queue<pair<int, int>> q; q.push({ans[s] = 0, s});
while (!q.empty()) if (auto [uu, u] = q.top(); q.pop(), !vis[u]++)
    for (ans[u] = min(ans[u], -uu); auto [vv, v] : edge[u])
        if (ans[u] + vv < ans[v])
            ans[v] = ans[u] + vv, q.push({-ans[v], v}), last[v] = u;

///dijstra(old)
auto dijstra = [&](int s)->vector<int> {
	vector<int> ans (n + 1, ~(1 << 31)), vis(n + 1);
	priority_queue <pair<int, int>> q; q.push({ans[s] = 0, s});
	while (!q.empty()) if (auto [pval, p] = q.top(); q.pop(), !vis[p]++) {
		for (ans[p] = min(ans[p], -pval); auto x : edge[p])
            if (ans[p] + x.val < ans[x.to])
				ans[x.to] = ans[p] + x.val, q.push({-ans[x.to], x.to});
	} return ans;
};

///floyd 全源最短路 O(n³)
int n, m; cin >> n >> m;
vector<vector<int>> f(n + 1, vector<int>(n + 1, 1 << 29));
for (int x = 1; x <= n; ++x) f[x][x] = 0;
for (int i = 0, j, k, l; i < m; ++i)
    cin >> j >> k >> l, if (l < f[j][k]) f[j][k] = f[k][j] = l;
for (int k = 1; k <= n; ++k)
    for (int x = 1; x <= n; ++x)
        for (int y = 1; y <= n; ++y)
            f[x][y] = min(f[x][y], f[x][k] + f[k][y]);         
for (int i =  1; i <= n; ++i, cout << endl)
    for (int j = 1; j <= n; ++j)
        cout << f[i][j] << ' ';

///bellman-ford 单源最短路 O(nm)
int n, m, s; cin >> n >> m >> s;
vector<array<long long, 3>> eg(m);
for (int i = 0; i < m; ++i)
    if (int j, k, l; cin >> j >> k >> l) eg[i] = {j, k, l};
vector<long long> dist(n + 1, (1ll << 31) - 1); dist[s] = 0;
for (int i = 1; i < n; ++i) 
    for (int j = 0; j < m; ++j)
        if (auto[u, v, w] = eg[j]; dist[u] + w < dist[v])
            dist[v] = dist[u] + w;
for (int i = 1; i <= n; ++i) cout << dist[i] << ' ';

///cvertex tarjan割点
vector<int> dfn(n + 1), low(n + 1);
auto tarjan = [&](auto dfs, int u, int fa) -> int {
    dfn[u] = low[u] = ++dfn[0];
    for (int cnt = 0; int v : edge[u]) low[0] = dfn[v],
        low[u] = min(low[u], low[0] ? dfn[v] : dfs(dfs, v, u)),
        cut[u] |= !low[0] && low[v] >= dfn[u] && (cnt+++fa);
    return low[u];
}; for (int i = 1; i <= n; ++i)
    !dfn[i] && tarjan(tarjan, i, 0);

///vdcc tarjan点双联通分量
vector<vector<int>> v-dcc;
vector<int> dfn(n + 1), low(n + 1), stk(n + 1);
auto tarjan = [&](auto dfs, int u, int fa) -> int {
    if (edge[u].empty() && !fa) vdcc.push_back(vector{u}); 
    for (dfn[u] = low[u] = ++dfn[0], stk[++stk[0]] = u;int v : edge[u]) 
        if (dfn[v]) low[u] = min(low[u], dfn[v]); else { 
            low[u] = min(low[u], dfs(dfs, v, u));
            if (vector<int> tmp; low[v] >= dfn[u]) {
                while (tmp.push_back(stk[stk[0]]), v != stk[stk[0]--]);
                tmp.push_back(u), vdcc.push_back(tmp);
            }
    } return low[u];
}; for (int i = 1; i <= n; ++i) 
    !dfn[i] && tarjan(tarjan, i, 0);

///cbridge tarjan割边
vector<pair<int, int>> bridge;
vector<int> dfn(n + 1), low(n + 1);
auto tarjan = [&](auto dfs, int u, int fa) -> int {
    for (dfn[u] = low[u] = ++dfn[0]; int v : edge[u]) if (v != fa) 
        if (dfn[v]) low[u] = min(low[u], dfn[v]); else
            if (low[u] = min(low[u], dfs(dfs, v, u)); low[v] > dfn[u]) 
                bridge.push_back({u, v});
    return low[u];
}; for (int i = 1; i <= n; ++i)
    !dfn[i] && tarjan(tarjan, i, 0);

///e-dcc tarjan边双联通分量
vector<int> dfn(n + 1), low(n + 1), stk(n + 1), edcc(n + 1);
auto tarjan = [&](auto dfs, int u, int fa) -> int {
    dfn[u] = low[u] = ++dfn[0], stk[++stk[0]] = u;
    for (int cnt = 0; int v : edge[u]) if (v != fa || cnt++)
        low[u] = min(low[u], dfn[v] ? dfn[v] : dfs(dfs, v, u));
    if (vector<int> tmp; low[u] == dfn[u] && ++low[0])
        while (edcc[stk[stk[0]]] = low[0], stk[stk[0]--] != u);
    return low[u];
}; for (int i = 1; i <= n; ++i)
    !dfn[i] && tarjan(tarjan, i, 0);    

///scc tarjan强双联通分量
vector<int> dfn(n + 1), low(n + 1), stk(n + 1), scc(n + 1), istk(n + 1);
auto tarjan = [&](auto dfs, int u, int fa) -> int {
    dfn[u] = low[u] = ++dfn[0], stk[++stk[0]] = u, istk[u] = 1;
    for (int cnt = 0; int v : edge[u]) 
        if (!dfn[v]) low[u] = min(low[u], dfs(dfs, v, u));
        else if(istk[v]) low[u] = min(low[u], dfn[v]);
    if (low[u] == dfn[u] && ++low[0]) 
        do scc[stk[stk[0]]] = low[0], istk[stk[stk[0]]] = 0;
        while (stk[stk[0]--] != u);
    return low[u];
}; for (int i = 1; i <= n; ++i)
    !dfn[i] && tarjan(tarjan, i, 0); 

///kruskl 最小生成树
vector<array<int, 3>> edge(m + 1);
for (int i = 1; i <= m; ++i) 
    cin >> edge[i][0] >> edge[i][1] >> edge[i][2];
sort(edge.begin(), edge.end(), [&](auto x, auto y) { return x[2] < y[2]; });
int ans = 0;
for (int i = 1; i <= m; ++i) 
    if (int x = get(edge[i][0]), y = get(edge[i][1]); x != y)
        fa[x] = y, ans += edge[i][2];

///mstr 最小斯坦纳树
vector dp(1 << k, vector(n + 1, 1 << 29));
for (int i = 0, j; i < k; ++i)
    cin >> dp[0][0], dp[1 << i][dp[0][0]] = 0;   
for (int s = 1; s < 1 << k; ++s) {
    priority_queue<pair<int, int>> q;
    for (int i = 1; i <= n; q.push({-dp[s][i], i}), ++i) 
        for (int t = s; t = (t - 1) & s;)
            dp[s][i] = min(dp[s][i], dp[t][i] + dp[s ^ t][i]);
    while (!q.empty()) if (auto [uu, u] = q.top(); q.pop(), -uu <= dp[s][u])
        for (auto [v, vv]: edge[u]) if (dp[s][u] + vv < dp[s][v])
            dp[s][v] = dp[s][u] + vv, q.push({-dp[s][v], v});
}
cout << dp[(1 << k) - 1][dp[0][0]];

///dsu 并查集
vector<int> fa(n + 1);
iota(fa.begin(), fa.end(), 0);
auto get = [&, g = [&](auto g, int x){
    if (x == fa[x]) return x;
    return fa[x] = g(g, fa[x]);
}](int x) { return g(g, x); };

///hungarian 二分图最大匹配 匈牙利算法
    vector<int> vis(m + 1), match(m + 1);
    auto hungarian = [&](auto dfs, int u)->bool {
        for (auto v : edge[u]) if (!vis[v]++)
            if (!match[v] || dfs(dfs, match[v]))
                return match[v] = u; return 0;
    }; for (int i = 1; i <= n; ++i)
        vis.assign(m + 1, 0), vis[0] += hungarian(hungarian, i);

///KM 二分图最大权完美匹配 O(N^3)
#define int long long
#define inf (1ll << 60)
signed main() {   _My_code
    int n, m; cin >> n >> m;
    vector<vector<pair<int, int>>> edge(n + 1);
    for (int i = 0; i < m; ++i)
        if (int j, k, l; cin >> j >> k >> l)
            edge[j].push_back({k, l});
    vector<int> match(n + 1);
    auto KM = [&]()->int {
        vector<int> A(n + 1, -inf), B(n + 1); 
        for (int i = 1; i <= n; ++i)
            for (auto [v, w] : edge[i]) A[i] = max(A[i], w);
        for (int i = 1, st; i <= n; ++i) {
            vector<int> va(n + 1), vb(n + 1), last(n + 1);
            vector<int> upd(n + 1, inf);
            auto hungarian = [&](auto dfs, int u, int fa)->bool { 
                for (++va[u]; auto [v, w] : edge[u]) if (!vb[v])
                    if (int det = A[u] + B[v] - w; det != 0) upd[v] > det && (upd[v] = det, last[v] = fa);
                    else if (++vb[v], last[v] = fa; !match[v] || dfs(dfs, match[v], v))
                        return match[v] = u, 1; return 0;
            };
            for (st = 0, match[st] = i; upd[0] = inf, match[st] && ! hungarian(hungarian, match[st], st); vb[st] = 1) {
                for (int v = 1; v <= n; ++v) if (!vb[v] && upd[v] < upd[0]) upd[0] = upd[v], st = v;
                for (int x = 1; x <= n; ++x) va[x] && (A[x] -= upd[0]), vb[x] ? B[x] += upd[0] : upd[x] -= upd[0];  
            } 
            while (st) match[st] = match[last[st]], st = last[st];
        }
        int ans = 0;
        for (int u = 1; u <= n; ++u) 
            for (auto [v, w] : edge[u])
                if (match[v] == u) ans += w;
        return ans;
    }; cout << KM() << endl;
    for (int i = 1; i <= n; ++i)
        cout << match[i] << ' ';
}
#undef int

///hlpp 最大流
#define int long long
signed main() { _My_code
    int n, m, s, t; cin >> n >> m >> s >> t;
    vector<vector<int>> eg(n + 1);
    vector<array<int, 2>> edge(m << 1); 
    int tot = 0;
    for (int i = 0; i < m; ++i)
        if (int j, k, l; cin >> j >> k >> l)
                edge[tot] = {k, l}, eg[j].push_back(tot++),
                edge[tot] = {j, 0}, eg[k].push_back(tot++);

    int level = 0, inf = 1ll << 60;
    stack<int> B[n];
    vector<int> ht(n + 1, inf), exflow(n + 1), gap(n);
    auto bfs = [&]() {
        queue<int> q; q.push(t), ht[t] = 0;
        while (q.size()) { int u = q.front(); q.pop();
            for (auto x : eg[u]) if (int v = edge[x][0], w = edge[x ^ 1][1]; w && ht[v] > ht[u] + 1)
                ht[v] = ht[u] + 1, q.push(v);
        } return ht[s] == inf;
    };
    auto push = [&](int u) {
        for (auto x : eg[u]) if (auto [v, w] = edge[x]; w && (ht[u] == ht[v] + 1 || u == s) && ht[v] <= n) {
            int d = u == s ? w : min(w, exflow[u]); 
            if (v != s && v != t && !exflow[v]) B[ht[v]].push(v), level = max(level, ht[v]);
            edge[x][1] -= d, edge[x ^ 1][1] += d; exflow[u] -= d, exflow[v] += d; 
            if (!exflow[u]) return 0;
        } return 1;
    };
    auto select = [&]() {
        while (~level && B[level].empty()) --level;
        if (int x; ~level) { x = B[level].top(); B[level].pop(); return x; }
        else return 0ll;
    };
    auto relabel = [&](int u) {
        for (ht[u] = inf; auto x : eg[u]) if (auto [v, w] = edge[x]; w) ht[u] = min(ht[u], ht[v]);
        if (++ht[u] < n) B[ht[u]].push(u), level = max(level, ht[u]), ++gap[ht[u]];
    };
    auto hlpp = [&]() {
        if (bfs()) return 0ll;
        for (int i = 1; i <= n; ++i)
            if (ht[i] != inf) ++gap[ht[i]];
        ht[s] = n; push(s); 
        for (int u; u = select();) if (push(u)) {
            if (!--gap[ht[u]]) for (int i = 1; i <= n; ++i)
                if (i != s && ht[i] > ht[u] && ht[i] < n + 1)
                    ht[i] = n + 1;
            relabel(u);
        } return exflow[t];
    }; cout << hlpp() << endl;
}
#undef int

///dinic 最大流
#define int long long
signed main() {   _My_code
    int n, m, s, t; cin >> n >> m >> s >> t;
    vector<vector<int>> eg(n + 1);
    vector<array<int, 2>> edge(m << 1); 
    int tot = 0;
    for (int i = 0; i < m; ++i)
        if (int j, k, l; cin >> j >> k >> l)
                edge[tot] = {k, l}, eg[j].push_back(tot++),
                edge[tot] = {j, 0}, eg[k].push_back(tot++);
    vector<int> dep(n + 1);
    auto bfs = [&]() {
        queue<int> q; q.push(s);
        dep.assign(n + 1, 0), dep[s] = 1;
        while (q.size()) { int u = q.front(); q.pop();
            for (auto x : eg[u]) if (auto [v, w] = edge[x]; !dep[v] && w)
                dep[v] = dep[u] + 1, q.push(v);
            } return dep[t];
    }; 
    auto dfs = [&](auto dfs, int u, int fl) {
        if (u == t || !fl) return fl;
        int sum = 0;
        for (auto x : eg[u]) if (auto[v, w] = edge[x]; w && dep[v] == dep[u] + 1)
            if (int c = dfs(dfs, v, min(fl, w)); sum += c, edge[x][1] -= c, edge[x ^ 1][1] += c, !(fl -= c)) break;
        if (sum == 0) dep[u] = 0; return sum;
    };
    int maxflow = 0;
    while (bfs()) maxflow += dfs(dfs, s, 1ll << 60);
    cout << maxflow << endl;
}
#undef int

///ekmaxflow 最大流
int n, m, s, t; cin >> n >> m >> s >> t;
vector<vector<array<int, 2>>> edge(n + 1); 
vector<vector<int>> idx(n + 1, vector<int>(n + 1 , -1));
for (int i = 0; i < m; ++i)
    if (int j, k, l; cin >> j >> k >> l) {
        if (~idx[j][k]) edge[j][idx[j][k]][1] += l;
        else idx[j][k] = edge[j].size(), edge[j].push_back({k, l});
        if (!~idx[k][j]) idx[k][j] = edge[k].size(), edge[k].push_back({j, 0});
    }

int maxflow;
vector<int> incf(n + 1), pre(n + 1);
auto bfs = [&]() {
    vector<int> vis(n + 1);
    queue<int> q; q.push(s), ++vis[s]; incf[s] = ~(1 << 31);
    while (q.size()) { int u = q.front(); q.pop();
        for (auto [v, w] : edge[u]) if (w && !vis[v]++) 
            if (q.push(v), incf[v] = min(incf[pre[v] = u], w); v == t) return 1;
    } return 0;
}; for (maxflow = 0; bfs(); maxflow += incf[t])
    for (int u = t, v; v = pre[u], u != s; u = v)
        edge[u][idx[u][v]][1] += incf[t], edge[v][idx[v][u]][1] -= incf[t];

///AVL_Tree 平衡树
template<class T>
struct AVL_tree  {
    vector<T> data;
    vector<array<int, 2>> ch;
    vector<int> sz, hi, fa, wt;
    int n = 0, hd = 0, tot = 0, u = 0;
    AVL_tree(int N) : n(N) { sz = hi = fa = wt = vector<int>(n + 1), data.resize(n + 1), ch.resize(n + 1); }
    void uph(int x, int y) { x == hd && (hd = y); }
    void upd(int x)        { hi[x] = max(hi[ch[x][0]], hi[ch[x][1]]) + 1, sz[x] = sz[ch[x][0]] + sz[ch[x][1]] + wt[x], wt[0] = hi[0] = sz[0] = 0; }
    void tie(int _fa, int _ch, bool lr) { ch[fa[_ch] = _fa][lr] = _ch, ch[0][0] = ch[0][1] = fa[0] = 0; }
    void rtt(int x, bool lr) { int y = ch[x][lr]; tie(x, ch[y][!lr], lr), tie(fa[x], y, inv(x)), tie(y, x, !lr), uph(x, y), upd(x), upd(y); }
    bool inv(int x) { return x == ch[fa[x]][1]; }
    int adj(int x, bool lr) {
        if (ch[x][lr]) for (x = ch[x][lr]; ch[x][!lr]; x = ch[x][!lr]);
        else           for (int _x = x; fa[x] && (lr ? data[x] <= data[_x] : data[x] >= data[_x]); x = fa[x]);
        return x; 
    }
    void update(int x) { 
        for (int dif, lr; x; x = fa[x])
            if (upd(x), dif = hi[ch[x][0]] - hi[ch[x][1]], lr = dif < 0; abs(dif) > 1) 
                hi[ch[ch[x][lr]][lr]] > hi[ch[ch[x][lr]][!lr]] ? rtt(x, lr) : (rtt(ch[x][lr], !lr), rtt(x, lr));
    }
    void add(T val) {
        if (u = hd; !hd) return hd = ++tot, data[hd] = val, ++wt[hd], upd(hd);
        while (val != data[u] && ch[u][val > data[u]]) u = ch[u][val > data[u]];
        if (val == data[u]) ++wt[u], update(u); 
        else tie(u, ++tot, val > data[u]), data[tot] = val, ++wt[tot], update(tot);
    }
    void del(int x) {
        if (wt[x] > 1) return --wt[x], update(x);
        if (!ch[x][1] || !ch[x][0]) return tie(fa[x], ch[x][!ch[x][0]], inv(x)), uph(x, ch[x][!ch[x][0]]), update(fa[x]); 
        if (int pre = adj(x, 0), pre_f = fa[pre]; uph(x, pre), pre_f == x) return tie(fa[x], pre, inv(x)), tie(pre, ch[x][1], 1), update(pre);
        else return tie(pre_f, ch[pre][0], 1), tie(fa[x], pre, inv(x)), tie(pre, ch[x][0], 0), tie(pre, ch[x][1], 1), update(pre_f);
    }
    int find(T val) {
        for (int u = hd; u; u = ch[u][val > data[u]]) 
            if (data[u] == val) return u; return 0;
    }
};

///AVL_Tree2
template<class T>
struct AVL_tree  {
    vector<T> data;
    vector<array<int, 2>> ch;
    vector<int> sz, hi, fa, wt;
    int n = 0, hd = 0, tot = 0, u = 0;
    AVL_tree(int N) : n(N) { sz = hi = fa = wt = vector<int>(n + 1), data.resize(n + 1), ch.resize(n + 1); }
    void uph(int x, int y) { x == hd && (hd = y); }
    void upd(int x)        { hi[x] = max(hi[ch[x][0]], hi[ch[x][1]]) + 1, sz[x] = sz[ch[x][0]] + sz[ch[x][1]] + wt[x], wt[0] = hi[0] = sz[0] = 0; }
    void tie(int _fa, int _ch, bool lr) { ch[fa[_ch] = _fa][lr] = _ch, ch[0][0] = ch[0][1] = fa[0] = 0; }
    void rtt(int x, bool lr) { int y = ch[x][lr]; tie(x, ch[y][!lr], lr), tie(fa[x], y, inv(x)), tie(y, x, !lr), uph(x, y), upd(x), upd(y); }
    bool inv(int x) { return x == ch[fa[x]][1]; }
    int adj(int x, bool lr) {
        if (ch[x][lr]) for (x = ch[x][lr]; ch[x][!lr]; x = ch[x][!lr]);
        else           for (int _x = x; fa[x] && (lr ? data[x] <= data[_x] : data[x] >= data[_x]); x = fa[x]);
        return         x; 
    }
    void update(int x) { 
        for (int dif, lr; x; x = fa[x])
            if (upd(x), dif = hi[ch[x][0]] - hi[ch[x][1]], lr = dif < 0; abs(dif) > 1) 
                hi[ch[ch[x][lr]][lr]] > hi[ch[ch[x][lr]][!lr]] ? rtt(x, lr) : (rtt(ch[x][lr], !lr), rtt(x, lr));
    }
    void add(T val) {
        if (u = hd; !hd) return hd = ++tot, data[hd] = val, ++wt[hd], upd(hd);
        while (val != data[u] && ch[u][val > data[u]]) u = ch[u][val > data[u]];
        if (val == data[u]) ++wt[u], update(u); 
        else tie(u, ++tot, val > data[u]), data[tot] = val, ++wt[tot], update(tot);
    }
    void del(int x) {
        if (wt[x] > 1) return --wt[x], update(x);
        if (!ch[x][1] or !ch[x][0]) return tie(fa[x], ch[x][!ch[x][0]], inv(x)), uph(x, ch[x][!ch[x][0]]), update(fa[x]); 
        if (int pre = adj(x, 0), pre_f = fa[pre]; uph(x, pre), pre_f == x) return tie(fa[x], pre, inv(x)), tie(pre, ch[x][1], 1), update(pre);
        else return tie(pre_f, ch[pre][0], 1), tie(fa[x], pre, inv(x)), tie(pre, ch[x][0], 0), tie(pre, ch[x][1], 1), update(pre_f);
    }
    int find(T val) {
        for (int u = hd; u; u = ch[u][val > data[u]]) 
            if (data[u] == val) return u; return 0;
    }
    int print(int flag = 0, int x = 0, int w = 0) {
        queue<int> q; q.push(x = x ? x : hd);
        for (int i = 0; i < hi[x]; ++i) {
            for (int j = 0; j < 1 << i; ++j) {
                int u = q.front(); q.pop();
                if (flag == 0) cout << setw(1 << (hi[x] - i) + w) << data[u];
                if (flag == 1) cout << setw(1 << (hi[x] - i) + w) << u;
                if (flag == 2) cout << setw(1 << (hi[x] - i) + w) << wt[u];
                cout << setw(1 << (hi[x] - i) + w) << ' ';
                q.push(ch[u][0]);
                q.push(ch[u][1]);
            }
            cout << endl;
        } cout << endl;
        return 0;
    }
    int idx (int val) {
        int u = hd, id = 0;
        while (u && data[u] != val)
            if (data[u] < val) id += sz[ch[u][0]] + wt[u], u = ch[u][1];
            else u = ch[u][0]; 
        if (data[u] == val) id += sz[ch[u][0]];
        return id;
    }
    int valadj(int val, bool lr) {
        for (u = hd; ch[u][val > data[u]] && data[u] != val; u = ch[u][val > data[u]]);
        if (data[u] == val) return adj(u, lr);
        else                return lr ^ data[u] < val ? u : adj(u, lr);  
    }
     int get(int m) {
        for (u = hd; 1;) 
            if (sz[ch[u][0]] <= m && sz[ch[u][0]] + wt[u] > m) return u;
            else if (sz[ch[u][0]] < m) m -= sz[ch[u][0]] + wt[u], u = ch[u][1];
            else                       u = ch[u][0];  
    }
};

///Segtree 线段树
template<class Info, class Tag>
struct Segtree {
#define ls (u << 1)
#define rs (u << 1 | 1)
#define mid ((s + t) >> 1)
    int n;
    vector<Tag> tag; 
    vector<Info> data;
    void apply(int u, Tag v, int s, int t) {
        data[u].apply(v, t - s + 1); tag[u].apply(v);
    }
    void push_up(int u) { 
        data[u] = data[ls] + data[rs]; 
    }
    void push_down(int u , int s, int t) {
        apply(ls, tag[u], s, mid), apply(rs, tag[u], mid + 1, t);
        tag[u] = Tag();
    }
    Segtree(int N, vector<long long>::iterator it) : n(N) { 
        data.assign(4 << __lg(n), Info());
        tag.assign(4 << __lg(n), Tag());
        auto build = [&](auto build, int s, int t, int u)->Info {
            if (s == t) return data[u] = Info(*(it + s - 1));
            build(build, s, mid, ls);
            build(build, mid + 1, t, rs);
            push_up(u); return Info();
        }; build(build, 1, n, 1);
    }
    void modify(int l, int r, Tag v) {
        auto modify = [&](auto modify, int s, int t, int u)->void {
            if (s >= l && t <= r) return apply(u, v, s, t);
            push_down(u, s, t);
            if (mid >= l) modify(modify, s, mid, ls); 
            if (mid < r) modify(modify, mid + 1, t, rs); 
            push_up(u);
        }; modify(modify, 1, n, 1);
    }
    Info query(int l, int r) {
        auto query = [&](auto query, int s, int t, int u)->Info {
            if (s >= l && t <= r) return data[u]; 
            push_down(u, s, t);
            if (mid >= r) return query(query, s, mid, ls);
            if (mid < l) return query(query, mid + 1, t, rs); 
            return query(query, s, mid, ls) + query(query, mid + 1, t, rs);
        }; return query(query, 1, n, 1);
    }
#undef ls
#undef rs
#undef mid
};
struct Tag {
    long long val;
    Tag (long long v = 0) : val(v) {}
    void apply(Tag &v) {
        val += v.val;
    }
};
struct Info {
    long long val;
    Info (long long v = 0) : val(v) {}
    void apply(Tag &v, int len) {
        val += v.val * len;
    }
    Info operator+(Info rhs) {
        return val + rhs.val;
    }
};

///Segtree(Old) 线段树
struct Segtree {
#define ls (u << 1)
#define rs (u << 1 | 1)
#define mid ((s + t) >> 1)
    vector<long long> data, mark; int n;
    Segtree(int N, vector<long long>::iterator it) : n(N) { 
        data = mark = vector<long long>(4 << __lg(n), 0);
        auto build = [&](auto build, int s, int t, int u)->void {
            if (s == t) data[u] = *(it + s - 1);
            else  build(build, s, mid, ls), build(build, mid + 1, t, rs), 
                  data[u] = data[ls] + data[rs];
        };  build(build, 1, n, 1);
    }
    void modify(int l, int r, long long val) {
        auto modify = [&](auto modify, int s, int t, int u)->long long {
            if (s >= l && t <= r) return mark[u] += val, data[u] += val * (t - s + 1);
            data[ls] += mark[u] * (mid - s + 1), data[rs] += mark[u] * (t - mid);
            mark[ls] += mark[u], mark[rs] += mark[u], mark[u] = 0;
            if (mid >= l) modify(modify, s, mid, ls); 
            if (mid < r) modify(modify, mid + 1, t, rs); 
            return data[u] = data[ls] + data[rs];
        }; modify(modify, 1, n, 1);
    }
    long long query(int l, int r) {
        auto query = [&](auto query, int s, int t, int u)->long long {
            if (s >= l && t <= r) return data[u]; 
            data[ls] += mark[u] * (mid - s + 1), data[rs] += mark[u] * (t - mid);
            mark[ls] += mark[u], mark[rs] += mark[u], mark[u] = 0;
            if (mid >= r) return query(query, s, mid, ls);
            if (mid < l) return query(query, mid + 1, t, rs); 
            return query(query, s, mid, ls) + query(query, mid + 1, t, rs);
        }; return query(query, 1, n, 1);
    }
#undef ls
#undef rs
#undef mid
};

///fenwick 树状数组
struct fenwick {
    int n;
    vector<long long> val;
    int lowbit(int x) { return x & -x; } 
    fenwick(int N, vector<long long>::iterator a) : n(N) {
        val.assign(n + 1, 0);      
        for (int i = 1; i <= n; ++i) 
            if (val[i] += *(a + i - 1); i + lowbit(i) <= n) val[i + lowbit(i)] += val[i];
    }
    void modify(int x, long long k) { while (x <= n) val[x] += k, x += lowbit(x); }
    long long query(int l, int r) {
        auto query = [&](int x) {
            long long ans = 0;
            while (x) ans += val[x], x -= lowbit(x);
            return ans;
        }; return query(r) - query(l - 1);
    }
};
