struct MaxFlow {
    struct _Edge {
        int to;
        ll cap;
        _Edge(int to, ll cap) : to(to), cap(cap) {}
    };

    int n, s, t, worked = 0;
    ll maxflow;
    vector<_Edge> e;
    vector<vector<int>> g;
    vector<int> cur, h;

    MaxFlow(int n, int s, int t) : n(n), s(s), t(t) {
        g.assign(n + 1, {});
        cur.resize(n + 1);
        h.resize(n + 1);
    }

    void add(int u, int v, ll cap) {
        g[u].push_back(e.size());
        e.emplace_back(v, cap);
        g[v].push_back(e.size());
        e.emplace_back(u, 0);
    }

    bool bfs() {
        h.assign(n + 1, 0);
        queue<int> q;
        h[s] = 1, q.push(s);
        while(!q.empty()) {
            const int u = q.front();
            q.pop();
            for(int i : g[u]) {
                auto [v, cap] = e[i];
                if(cap > 0 && !h[v]) {
                    h[v] = h[u] + 1;
                    if(v == t) return true;
                    q.push(v);
                }
            }
        }
        return false;
    }

    ll dfs(int u, ll flow) {
        if(u == t) return flow;
        ll rest = flow;
        for(int& i = cur[u]; i < (int)g[u].size(); ++i) {
            const int j = g[u][i];
            auto [v, cap] = e[j];
            if(cap > 0 && h[v] == h[u] + 1) {
                ll cost = dfs(v, min(rest, cap));
                e[j].cap -= cost;
                e[j ^ 1].cap += cost;
                rest -= cost;
                if(rest == 0) return flow;
            }
        }
        return flow - rest;
    }

    ll operator()() {
        if(worked) return maxflow;
        ll ans = 0;
        while(bfs()) {
            cur.assign(n + 1, 0);
            ans += dfs(s, INF);
        }
        worked = 1;
        return maxflow = ans;
    }

    // cut[U] is True  if U is in the S-set
    // cut[U] is False if U is in the T-set
    vector<bool> minCut() {
        (*this)();
        vector<bool> cut(n + 1);
        for(int i = 1; i <= n; ++i) {
            cut[i] = h[i] != 0;
        }
        return cut;
    }

    struct Edge {
        int from, to;
        ll cap, flow;
    };
    vector<Edge> edges() {
        vector<Edge> res;
        for(int i = 0; i < (int)e.size(); i += 2) {
            Edge x;
            x.from = e[i + 1].to;
            x.to = e[i].to;
            x.cap = e[i].cap + e[i + 1].cap;
            x.flow = e[i + 1].cap;
            res.push_back(x);
        }
        return res;
    }
};
