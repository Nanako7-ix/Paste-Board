auto dijkstra = [&](auto& adj, int s) -> vector<ll> {
    int n = adj.size() - 1;
    vector<int> vis(n + 1);
    vector<ll> dp(n + 1, INF);
    priority_queue<pair<int, ll>, vector<pair<int, ll>>, greater<pair<int, ll>>> q;
    dp[s] = 0, q.push({0, s});
    while(!q.empty()) {
        auto [d, u] = q.top();
        q.pop();
        if(vis[u]) continue;
        vis[u] = 1;
        for(auto [v, w] : adj[u]) {
            if(dp[v] > dp[u] + w) {
                dp[v] = dp[u] + w;
                q.push({dp[v], v});
            }
        }
    }
    return dp;
};
