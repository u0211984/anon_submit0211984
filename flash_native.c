#include <stdlib.h>
#include <string.h>

#define R_L2 0
#define R_HBM 1
#define R_PER_SM 13
#define R_SMEM(sm) (2 + (sm) * R_PER_SM)
#define R_TENSOR(sm, w) (2 + (sm) * R_PER_SM + 1 + (w))
#define R_VECTOR(sm, w) (2 + (sm) * R_PER_SM + 5 + (w))
#define R_SFU(sm, w) (2 + (sm) * R_PER_SM + 9 + (w))
#define R_NONE (-1)

typedef struct {
    double ready;
    double service;
    double duration;
    int resource;
    int owner;
    int pending;
    int refs;
    int nchild, cchild;
    int *children;
    void *ctx;
    unsigned char cb, is_done;
    int next_free;
} Node;

#define OWNER_LIMIT (1LL << 31)
typedef struct {
    double t;
    unsigned long long serial;
    unsigned int hi;
    int waiter;
    int node;
} Event;

typedef struct {
    int owner;
    long long number;
    int waiter;
} OrderItem;

typedef struct {
    int node;
    int res;
    long long number;
    double arrival_time;
    long long arrival_serial;
    long long first, end;
    int queued;
    int next_free;
} Waiter;

typedef struct {
    OrderItem *order;
    int norder, corder;
    double *acquired;
    long long nacq, cacq;
    long long arrivals;
    int exists;
} Waiters;

typedef struct {
    Event *h;
    long long n, cap;
} Heap;

struct Sim;
typedef void (*Callback)(struct Sim *, int node);

typedef struct Sim {
    Node *nodes;
    int nnodes, cnodes, free_node, live;
    Waiter *waiters;
    int nwaiters, cwaiters, free_waiter;
    Heap later;
    Heap now_queue;
    double *busy;
    double *service;
    long long *first_take;
    long long takes;
    Waiters *waiting;
    int nres;
    long long serial;
    double now;
    Callback *callbacks;
    int failed;

    const void *program;

    int jump_kind;
    void **slots;
    int nslots;
    int snap_due;
    struct Jump *jump;
    struct FF *ff;
    struct FFW *ffw;
} Sim;

static void *grow(void *p, long long *cap, long long need, size_t size, Sim *s) {
    if (need <= *cap) return p;
    long long c = *cap ? *cap : 16;
    while (c < need) c *= 2;
    void *q = realloc(p, (size_t)c * size);
    if (!q) { s->failed = 1; return p; }
    *cap = c;
    return q;
}

static int new_node(Sim *s) {
    int id;
    if (s->free_node >= 0) {
        id = s->free_node;
        s->free_node = s->nodes[id].next_free;
    } else {
        if (s->nnodes == s->cnodes) {
            long long c = s->cnodes;
            s->nodes = grow(s->nodes, &c, s->nnodes + 1, sizeof(Node), s);
            if (s->failed) return -1;
            memset(s->nodes + s->cnodes, 0, (size_t)(c - s->cnodes) * sizeof(Node));
            s->cnodes = (int)c;
        }
        id = s->nnodes++;
    }
    s->live++;
    return id;
}

static void free_node(Sim *s, int id) {
    Node *n = &s->nodes[id];
    n->is_done = 2;
    n->next_free = s->free_node;
    n->nchild = 0;
    n->ctx = NULL;
    s->free_node = id;
    s->live--;
}

static void hold(Sim *s, int id) { if (id >= 0) s->nodes[id].refs++; }

static void release(Sim *s, int id) {
    if (id < 0) return;
    Node *n = &s->nodes[id];
    if (--n->refs == 0 && n->is_done) free_node(s, id);
}

static int waiter_lt(Sim *s, int x, int y) {

    Waiter *a = &s->waiters[x], *b = &s->waiters[y];
    if (a->res == b->res) return a->number < b->number;
    const double *mine = s->waiting[a->res].acquired, *theirs = s->waiting[b->res].acquired;
    long long i = a->end - 1, j = b->end - 1;
    for (;;) {
        int my_arrival = i < a->first, their_arrival = j < b->first;
        double u = my_arrival ? a->arrival_time : mine[i];
        double v = their_arrival ? b->arrival_time : theirs[j];
        if (u != v) return u < v;
        if (my_arrival || their_arrival) {
            if (my_arrival && their_arrival) return a->arrival_serial < b->arrival_serial;
            return their_arrival;
        }
        i--;
        j--;
    }
}

static inline int event_lt(Sim *s, const Event *a, const Event *b) {

    if (a->t != b->t) return a->t < b->t;
    if (a->hi != b->hi) return a->hi < b->hi;
    if (a->serial != b->serial) return a->serial < b->serial;
    return waiter_lt(s, a->waiter, b->waiter);
}

static void ev_siftdown(Sim *s, Heap *q, long long startpos, long long pos) {
    Event *h = q->h;
    Event newitem = h[pos];
    while (pos > startpos) {
        long long parentpos = (pos - 1) >> 1;
        if (event_lt(s, &newitem, &h[parentpos])) {
            h[pos] = h[parentpos];
            pos = parentpos;
            continue;
        }
        break;
    }
    h[pos] = newitem;
}

static void ev_siftup(Sim *s, Heap *q, long long pos) {
    Event *h = q->h;
    long long endpos = q->n, startpos = pos, limit = endpos >> 1;
    Event newitem = h[pos];
    while (pos < limit) {
        long long childpos = 2 * pos + 1;
        if (childpos + 1 < endpos && !event_lt(s, &h[childpos], &h[childpos + 1]))
            childpos += 1;
        h[pos] = h[childpos];
        pos = childpos;
    }
    h[pos] = newitem;
    ev_siftdown(s, q, startpos, pos);
}

static void ev_push(Sim *s, double t, int type, int owner, long long serial, int waiter, int node) {
    Heap *q = t == s->now ? &s->now_queue : &s->later;
    if (q->n == q->cap) {
        q->h = grow(q->h, &q->cap, q->n + 1, sizeof(Event), s);
        if (s->failed) return;
    }
    if (owner < 0 || (long long)owner >= OWNER_LIMIT || serial < 0) { s->failed = 3; return; }
    Event *e = &q->h[q->n++];
    e->t = t;
    e->hi = ((unsigned int)type << 31) | (unsigned int)owner;
    e->serial = (unsigned long long)(waiter >= 0 ? 0 : serial);
    e->waiter = waiter;
    e->node = node;
    ev_siftdown(s, q, 0, q->n - 1);
}

static Event heap_pop(Sim *s, Heap *q) {
    Event last = q->h[--q->n];
    if (q->n) {
        Event ret = q->h[0];
        q->h[0] = last;
        ev_siftup(s, q, 0);
        return ret;
    }
    return last;
}

static Event ev_pop(Sim *s) {
    if (!s->now_queue.n) return heap_pop(s, &s->later);
    if (s->later.n && event_lt(s, &s->later.h[0], &s->now_queue.h[0])) return heap_pop(s, &s->later);
    return heap_pop(s, &s->now_queue);
}

static int order_lt(const OrderItem *a, const OrderItem *b) {
    if (a->owner != b->owner) return a->owner < b->owner;
    return a->number < b->number;
}

static void ord_siftdown(OrderItem *h, int startpos, int pos) {
    OrderItem newitem = h[pos];
    while (pos > startpos) {
        int parentpos = (pos - 1) >> 1;
        if (order_lt(&newitem, &h[parentpos])) {
            h[pos] = h[parentpos];
            pos = parentpos;
            continue;
        }
        break;
    }
    h[pos] = newitem;
}

static void ord_siftup(OrderItem *h, int endpos, int pos) {
    int startpos = pos, limit = endpos >> 1;
    OrderItem newitem = h[pos];
    while (pos < limit) {
        int childpos = 2 * pos + 1;
        if (childpos + 1 < endpos && !order_lt(&h[childpos], &h[childpos + 1]))
            childpos += 1;
        h[pos] = h[childpos];
        pos = childpos;
    }
    h[pos] = newitem;
    ord_siftdown(h, startpos, pos);
}

static void ord_push(Sim *s, Waiters *w, OrderItem item) {
    long long c = w->corder;
    w->order = grow(w->order, &c, w->norder + 1, sizeof(OrderItem), s);
    if (s->failed) return;
    w->corder = (int)c;
    w->order[w->norder++] = item;
    ord_siftdown(w->order, 0, w->norder - 1);
}

static void ord_pop(Waiters *w) {
    OrderItem last = w->order[--w->norder];
    if (w->norder) {
        w->order[0] = last;
        ord_siftup(w->order, w->norder, 0);
    }
}

struct Jump;
struct FF;
static __thread long long ff_stats[2];
static void ff_take(Sim *s, int owner, int resource, double service);
struct FFW;
static void ffw_take(Sim *s, int owner, int resource, double service);
#define JUMP_FWD 1
#define JUMP_BWD 2
static void jump_note_values(Sim *s, double service, double duration);
static void jump_point(Sim *s);
static int jump_init(Sim *s, int kind, int sms, int resident);

static int add(Sim *s, const int *deps, int ndeps, int resource, double service, double latency, int owner,
               int cb, void *ctx) {
    double ready = s->now;
    int id = new_node(s);
    if (id < 0) return -1;
    Node *node = &s->nodes[id];
    node->resource = resource;
    node->service = service;
    node->duration = latency > service ? latency : service;
    if (s->jump) jump_note_values(s, service, node->duration);
    node->owner = owner;
    node->cb = (unsigned char)cb;
    node->ctx = ctx;
    node->is_done = 0;
    node->refs = 0;
    node->nchild = 0;
    int pending = 0;
    for (int i = 0; i < ndeps; i++) {
        Node *dep = &s->nodes[deps[i]];
        if (dep->is_done) {
            if (dep->ready > ready) ready = dep->ready;
        } else {
            if (dep->nchild == dep->cchild) {
                long long c = dep->cchild;
                dep->children = grow(dep->children, &c, dep->nchild + 1, sizeof(int), s);
                if (s->failed) return -1;
                dep->cchild = (int)c;
            }
            dep->children[dep->nchild++] = id;
            pending++;
        }
    }
    node = &s->nodes[id];
    node->ready = ready;
    node->pending = pending;
    if (pending == 0) {
        s->serial++;
        ev_push(s, ready, 1, owner, s->serial, -1, id);
    }
    return id;
}

static int new_waiter(Sim *s) {
    int id;
    if (s->free_waiter >= 0) {
        id = s->free_waiter;
        s->free_waiter = s->waiters[id].next_free;
    } else {
        long long c = s->cwaiters;
        s->waiters = grow(s->waiters, &c, s->nwaiters + 1, sizeof(Waiter), s);
        if (s->failed) return -1;
        s->cwaiters = (int)c;
        id = s->nwaiters++;
    }
    return id;
}

static double run(Sim *s) {
    while ((s->now_queue.n || s->later.n) && !s->failed) {
        Event ev = ev_pop(s);
        double now = ev.t;
        int owner = (int)(ev.hi & 0x7fffffffu);
        long long ev_serial = (long long)ev.serial;
        int id = ev.node;
        s->now = now;
        if (ev.hi >> 31) {
            Node *node = &s->nodes[id];
            int resource = node->resource;
            if (resource == R_NONE) {
                s->serial++;
                double duration = node->duration;
                ev_push(s, now + duration, 0, owner, s->serial, -1, id);
                continue;
            }
            Waiters *waiters = s->waiting[resource].exists ? &s->waiting[resource] : NULL;
            if (ev.waiter >= 0) {
                int token = ev.waiter;
                if (s->busy[resource] > now) {
                    if (waiters->order[0].waiter == token) {
                        s->waiters[token].end = waiters->nacq;
                        ev_push(s, s->busy[resource], 1, owner, 0, token, id);
                    } else {
                        s->waiters[token].queued = 0;
                    }
                    continue;
                }
                if (waiters->order[0].waiter != token) { s->failed = 2; return 0.0; }
                ord_pop(waiters);
                s->waiters[token].node = -1;
                s->waiters[token].next_free = s->free_waiter;
                s->free_waiter = token;
            } else if (s->busy[resource] > now) {
                s->serial++;
                if (!waiters) {
                    waiters = &s->waiting[resource];
                    waiters->exists = 1;
                }
                waiters->arrivals++;
                int w = new_waiter(s);
                if (w < 0) return 0.0;
                Waiter *wt = &s->waiters[w];
                wt->node = id; wt->res = resource; wt->number = waiters->arrivals; wt->arrival_time = now;
                wt->arrival_serial = ev_serial; wt->first = waiters->nacq; wt->end = waiters->nacq; wt->queued = 0;
                OrderItem item = {owner, waiters->arrivals, w};
                ord_push(s, waiters, item);
                if (waiters->order[0].waiter == w) {
                    s->waiters[w].queued = 1;
                    ev_push(s, s->busy[resource], 1, owner, 0, w, id);
                }
                continue;
            }
            node = &s->nodes[id];
            double free = now + node->service;
            s->busy[resource] = free;
            if (s->first_take[resource] < 0) s->first_take[resource] = s->takes++;
            s->service[resource] = s->service[resource] + node->service;
            if (s->ff) ff_take(s, owner, resource, node->service);
            if (s->ffw) ffw_take(s, owner, resource, node->service);
            s->serial++;
            double duration = node->duration;
            ev_push(s, now + duration, 0, owner, s->serial, -1, id);
            if (waiters) {
                if (waiters->norder) {
                    if (free > now) {
                        waiters->acquired = grow(waiters->acquired, &waiters->cacq, waiters->nacq + 1,
                                                 sizeof(double), s);
                        if (s->failed) return 0.0;
                        waiters->acquired[waiters->nacq++] = now;
                        s->serial += waiters->norder;
                    }
                    int head = waiters->order[0].waiter;
                    if (!s->waiters[head].queued) {
                        s->waiters[head].queued = 1;
                        s->waiters[head].end = waiters->nacq;
                        int hn = s->waiters[head].node;
                        ev_push(s, free, 1, s->nodes[hn].owner, 0, head, hn);
                    }
                } else if (waiters->nacq) {
                    waiters->nacq = 0;
                }
            }
        } else {
            Node *node = &s->nodes[id];
            node->ready = now;
            node->is_done = 1;
            for (int i = 0; i < node->nchild; i++) {
                Node *child = &s->nodes[s->nodes[id].children[i]];
                int pending = child->pending - 1;
                child->pending = pending;
                if (now > child->ready) child->ready = now;
                if (pending == 0) {
                    s->serial++;
                    ev_push(s, child->ready, 1, child->owner, s->serial, -1, s->nodes[id].children[i]);
                }
            }
            node = &s->nodes[id];
            node->nchild = 0;
            if (node->cb) s->callbacks[node->cb](s, id);
            node = &s->nodes[id];
            if (node->refs == 0) free_node(s, id);
            if (s->snap_due) jump_point(s);
        }
    }
    return s->now;
}

static long long test_serial0 = 0;
void flash_native_test_set_serial0(long long value) { test_serial0 = value; }

static int sim_init(Sim *s, int sms) {
    memset(s, 0, sizeof(*s));
    s->serial = test_serial0;
    s->free_node = -1;
    s->free_waiter = -1;
    s->nres = 2 + sms * R_PER_SM;
    s->busy = calloc((size_t)s->nres, sizeof(double));
    s->service = calloc((size_t)s->nres, sizeof(double));
    s->first_take = malloc((size_t)s->nres * sizeof(long long));
    s->waiting = calloc((size_t)s->nres, sizeof(Waiters));
    if (!s->busy || !s->service || !s->first_take || !s->waiting) return -1;
    for (int i = 0; i < s->nres; i++) s->first_take[i] = -1;
    return 0;
}

static void jump_free(Sim *s);
static void ff_free(Sim *s);
static void ffw_free(Sim *s);
static int ff_init(Sim *s, int sms, int resident, int ncta, int n, double rtol);

static void sim_free(Sim *s) {
    ff_free(s);
    ffw_free(s);
    jump_free(s);
    for (int i = 0; i < s->nnodes; i++) free(s->nodes[i].children);
    free(s->nodes);
    free(s->waiters);
    free(s->later.h);
    free(s->now_queue.h);
    free(s->busy);
    free(s->service);
    free(s->first_take);
    if (s->waiting) {
        for (int i = 0; i < s->nres; i++) { free(s->waiting[i].order); free(s->waiting[i].acquired); }
    }
    free(s->waiting);
}

static int sim_output(Sim *s, double *service_out, long long *first_take_out) {
    for (int i = 0; i < s->nres; i++) { service_out[i] = s->service[i]; first_take_out[i] = s->first_take[i]; }
    return s->failed ? -s->failed : s->live;
}

typedef struct { int v[64]; int n; } Deps;
#define D0(d) ((d).n = 0)
static inline void dpush(Deps *d, int x) { d->v[d->n++] = x; }
static inline void dcopy(Deps *to, const Deps *from) { to->n = from->n; memcpy(to->v, from->v, (size_t)from->n * sizeof(int)); }

typedef struct {
    int sms, resident, ncta, warps, m, n, d, q_len, kv_len, dao, output_element_bytes, has_ranges;
    double smem_rate, l2_rate, hbm_rate, tensor_rate, vector_rate, sfu_rate;
    double mma_latency, smem_latency, l2_latency, hbm_latency, hbm_fraction;
    const long long *tiles;
    const long long *ranges;
    double *starts, *finishes;
    int next_cta;
} Fwd;

typedef struct {
    int cid, sm, slot;
    long long ntiles;
    long long remaining;
} CtaHead;

typedef struct {
    int cid, sm, slot;
    long long ntiles, remaining;
    long long nt, n_start, n_end;
    int valid_q;
    int qload;
    long long cur;
    struct FwdNext *pend;
} FwdCta;

typedef struct FwdNext {
    FwdCta *cta;
    long long index;
    int next_k;
} FwdNext;

enum { CB_NONE, CB_FWD_NEXT, CB_FWD_EPILOGUE, CB_FWD_COMPLETE, CB_BWD_FINISH, CB_MAIN_NEXT, CB_MAIN_EPILOGUE };

static void fwd_launch(Sim *s, int sm, int slot);
static int ffw_tile(Sim *s, FwdCta *c, long long index, long long *tiles, double *delay);
static void ffw_launch(Sim *s, FwdCta *c);
static int ffw_init(Sim *s, int sms, int resident, int ncta, int n, double rtol);
static int ff_tile(Sim *s, CtaHead *h, long long *tiles, double *delay);
static void ff_launch(Sim *s, CtaHead *h);
static inline double fabs_(double x) { return x < 0 ? -x : x; }

static int fwd_node(Sim *s, FwdCta *c, const Deps *d, int resource, double service, double latency, int cb,
                    void *ctx) {
    return add(s, d->v, d->n, resource, service, latency, c->cid, cb, ctx);
}

static int fwd_smem(Sim *s, FwdCta *c, double count, const Deps *d) {
    const Fwd *p = s->program;
    return fwd_node(s, c, d, R_SMEM(c->sm), count / p->smem_rate, count ? p->smem_latency : 0, 0, NULL);
}

static int fwd_load(Sim *s, FwdCta *c, double count, const Deps *d) {
    const Fwd *p = s->program;
    double l2_count = count;
    int l2 = fwd_node(s, c, d, R_L2, l2_count / p->l2_rate, l2_count ? p->l2_latency : 0, 0, NULL);
    int hbm = fwd_node(s, c, d, R_HBM, count * p->hbm_fraction / p->hbm_rate,
                       (count && p->hbm_fraction) ? p->hbm_latency : 0, 0, NULL);
    Deps x; D0(x); dpush(&x, l2); dpush(&x, hbm);
    return fwd_smem(s, c, count, &x);
}

static int fwd_vector(Sim *s, FwdCta *c, double count, int warp, const Deps *d) {
    const Fwd *p = s->program;
    return fwd_node(s, c, d, R_VECTOR(c->sm, warp % 4), count / p->vector_rate, 0.0, 0, NULL);
}

static int fwd_sfu(Sim *s, FwdCta *c, double count, int warp, const Deps *d) {
    const Fwd *p = s->program;
    return fwd_node(s, c, d, R_SFU(c->sm, warp % 4), count / p->sfu_rate, 0.0, 0, NULL);
}

static int fwd_gemm(Sim *s, FwdCta *c, int warp, const Deps *deps, int qk) {
    const Fwd *p = s->program;
    long long m = p->m, n = p->n, d = p->d, warps = p->warps;
    long long inner = qk ? d : n, cols = qk ? n : d;
    long long operands = (qk ? (m / warps + cols) : cols) * 16 * 2;
    double load_service = (double)operands / p->smem_rate;
    double load_latency = operands ? p->smem_latency : 0;
    double math_service = (double)(2 * (m / warps) * cols * 16) / p->tensor_rate;
    int tensor_resource = R_TENSOR(c->sm, warp % 4);
    int smem_resource = R_SMEM(c->sm);
    int last_load = -1, last_math = -1, prev_math = -1;
    for (long long group = 0; group < inner / 16; group++) {
        Deps pre; dcopy(&pre, deps);
        if (last_load >= 0) dpush(&pre, last_load);
        if (group > 1) dpush(&pre, prev_math);
        int ld = add(s, pre.v, pre.n, smem_resource, load_service, load_latency, c->cid, 0, NULL);
        int md[2]; int nm = 0;
        md[nm++] = ld;
        if (last_math >= 0) md[nm++] = last_math;
        int mm = add(s, md, nm, tensor_resource, math_service, p->mma_latency, c->cid, 0, NULL);
        last_load = ld;
        prev_math = last_math;
        last_math = mm;
    }
    return last_math;
}

static void fwd_epilogue(Sim *s, FwdCta *c, const Deps *deps_in) {
    const Fwd *p = s->program;
    long long m = p->m, d = p->d, warps = p->warps;
    Deps deps = *deps_in;
    if (c->nt) {
        Deps normed; D0(normed);
        for (int w = 0; w < warps; w++) {
            int reduction = fwd_vector(s, c, (double)(8 * m) / (double)warps, w, &deps);
            Deps r; D0(r); dpush(&r, reduction);
            int reciprocal = fwd_sfu(s, c, (double)m / (double)warps, w, &r);
            int logarithm = fwd_sfu(s, c, (double)m / (double)warps, w, &r);
            Deps rl; D0(rl); dpush(&rl, reciprocal); dpush(&rl, logarithm);
            dpush(&normed, fwd_vector(s, c, (double)(m * d + 2 * m) / (double)warps, w, &rl));
        }
        int staged = fwd_smem(s, c, (double)(m * d * p->output_element_bytes), &normed);
        Deps st; D0(st); dpush(&st, staged);
        int loaded = fwd_smem(s, c, (double)(m * d * p->output_element_bytes), &st);
        D0(deps); dpush(&deps, loaded);
    }
    long long stored = (long long)c->valid_q * (d * p->output_element_bytes + 4);
    double l2_count = (double)stored, hbm_count = (double)stored;
    int l2 = fwd_node(s, c, &deps, R_L2, l2_count / p->l2_rate, l2_count ? p->l2_latency : 0, 0, NULL);
    int hbm = fwd_node(s, c, &deps, R_HBM, hbm_count / p->hbm_rate, hbm_count ? p->hbm_latency : 0, 0, NULL);
    Deps fin; D0(fin); dpush(&fin, l2); dpush(&fin, hbm);
    fwd_node(s, c, &fin, R_NONE, 0.0, 0.0, CB_FWD_COMPLETE, c);
}

static void fwd_tile(Sim *s, FwdCta *c, long long index, const Deps *prev, int ready_k) {
    const Fwd *p = s->program;
    long long m = p->m, n = p->n, d = p->d, warps = p->warps;
    long long block = c->n_end - 1 - index;
    long long kv_rows = n < p->kv_len - block * n ? n : p->kv_len - block * n;
    c->ntiles++;
    c->remaining = block - c->n_start;
    c->cur = index;
    if (s->jump && c->sm == 0 && c->slot == 0) s->snap_due = 1;
    long long ff_r; double ff_d;
    if (s->ffw && index > 0 && ffw_tile(s, c, index, &ff_r, &ff_d)) {

        int skip = fwd_node(s, c, prev, R_NONE, 0.0, ff_d, 0, NULL);
        FwdNext *next = malloc(sizeof(FwdNext));
        if (!next) { s->failed = 1; return; }
        next->cta = c; next->index = index + ff_r; next->next_k = ready_k;
        hold(s, ready_k);
        c->pend = next;
        Deps z; D0(z); dpush(&z, skip);
        fwd_node(s, c, &z, R_NONE, 0.0, 0.0, CB_FWD_NEXT, next);
        return;
    }
    Deps b = *prev; dpush(&b, c->qload); dpush(&b, ready_k);
    int begin = fwd_node(s, c, &b, R_NONE, 0.0, 0.0, 0, NULL);
    Deps bg; D0(bg); dpush(&bg, begin);
    int vload = fwd_load(s, c, (double)(kv_rows * d * 2), &bg);
    int vzero = kv_rows < n ? fwd_smem(s, c, (double)((n - kv_rows) * d * 2), &bg) : vload;
    Deps bar; D0(bar);
    for (int w = 0; w < warps; w++) dpush(&bar, fwd_gemm(s, c, w, &bg, 1));
    dpush(&bar, vload); dpush(&bar, vzero);
    int barrier = fwd_node(s, c, &bar, R_NONE, 0.0, 0.0, 0, NULL);
    Deps br; D0(br); dpush(&br, barrier);
    int next_k = -1;
    if (block > c->n_start) next_k = fwd_load(s, c, (double)(n * d * 2), &br);
    Deps pv; D0(pv);
    for (int w = 0; w < warps; w++) {
        double row_lanes = (double)(4 * m) / (double)warps;
        double max_ops = (double)(m * n + (index == 0 ? 4 * m : 8 * m)) / (double)warps;
        int reduction = fwd_vector(s, c, max_ops, w, &br);
        Deps x; D0(x);
        if (index) {
            dpush(&x, reduction);
            int alpha_args = fwd_vector(s, c, 2 * row_lanes, w, &x);
            D0(x); dpush(&x, alpha_args);
            int alpha = fwd_sfu(s, c, row_lanes, w, &x);
            D0(x); dpush(&x, alpha);
            reduction = fwd_vector(s, c, row_lanes + (double)(m * d) / (double)warps, w, &x);
        }
        D0(x); dpush(&x, reduction);
        int scaled_max = fwd_vector(s, c, row_lanes, w, &x);
        D0(x); dpush(&x, scaled_max);
        int score = fwd_vector(s, c, (double)(m * n) / (double)warps * (double)(p->dao ? 2 : 1), w, &x);
        D0(x); dpush(&x, score);
        int exp = fwd_sfu(s, c, (double)(m * n) / (double)warps, w, &x);
        D0(x); dpush(&x, exp);
        int summed = fwd_vector(s, c, (double)(m * n - (index == 0 ? 4 * m : 0)) / (double)warps, w, &x);
        D0(x); dpush(&x, summed);
        dpush(&pv, fwd_gemm(s, c, w, &x, 0));
    }
    if (block > c->n_start) {
        FwdNext *next = malloc(sizeof(FwdNext));
        if (!next) { s->failed = 1; return; }
        next->cta = c; next->index = index + 1; next->next_k = next_k;
        hold(s, next_k);
        c->pend = next;
        fwd_node(s, c, &pv, R_NONE, 0.0, 0.0, CB_FWD_NEXT, next);
    } else {
        fwd_node(s, c, &pv, R_NONE, 0.0, 0.0, CB_FWD_EPILOGUE, c);
    }
}

static void cb_fwd_next(Sim *s, int id) {
    FwdNext *next = s->nodes[id].ctx;
    next->cta->pend = NULL;
    Deps prev; D0(prev); dpush(&prev, id);
    fwd_tile(s, next->cta, next->index, &prev, next->next_k);
    release(s, next->next_k);
    free(next);
}

static void cb_fwd_epilogue(Sim *s, int id) {
    Deps prev; D0(prev); dpush(&prev, id);
    fwd_epilogue(s, s->nodes[id].ctx, &prev);
}

static void cb_fwd_complete(Sim *s, int id) {
    Fwd *p = (Fwd *)s->program;
    FwdCta *c = s->nodes[id].ctx;
    p->finishes[c->cid] = s->now;
    int sm = c->sm, slot = c->slot;
    if (s->slots) s->slots[slot * p->sms + sm] = NULL;
    release(s, c->qload);
    free(c);
    fwd_launch(s, sm, slot);
}

static void fwd_launch(Sim *s, int sm, int slot) {
    Fwd *p = (Fwd *)s->program;
    if (p->next_cta == p->ncta) return;
    int cid = p->next_cta++;
    p->starts[cid] = s->now;
    long long qblocks = (p->q_len + p->m - 1) / p->m;
    long long mb = cid % qblocks;
    long long vq = p->q_len - mb * p->m;
    FwdCta *c = malloc(sizeof(FwdCta));
    if (!c) { s->failed = 1; return; }
    c->cid = cid; c->sm = sm; c->slot = slot;
    c->ntiles = 0; c->remaining = 0; c->cur = 0; c->pend = NULL;
    if (s->slots) s->slots[slot * p->sms + sm] = c;
    if (s->ffw) ffw_launch(s, c);
    c->valid_q = (int)(p->m < vq ? p->m : vq);
    c->nt = p->tiles[cid];
    if (p->has_ranges) { c->n_start = p->ranges[2 * cid]; c->n_end = p->ranges[2 * cid + 1]; }
    else { c->n_start = 0; c->n_end = c->nt; }
    c->qload = -1;
    Deps none; D0(none);
    if (c->nt == 0) {
        fwd_epilogue(s, c, &none);
        return;
    }
    c->qload = fwd_load(s, c, (double)((long long)c->valid_q * p->d * 2), &none);
    hold(s, c->qload);
    long long last_n = c->n_end - 1;
    long long rows = p->n < p->kv_len - last_n * p->n ? p->n : p->kv_len - last_n * p->n;
    int kload = fwd_load(s, c, (double)(rows * p->d * 2), &none);
    fwd_tile(s, c, 0, &none, kload);
}

int flash_native_forward(const long long *ints, const double *reals, const long long *tiles,
                         const long long *ranges, double *out_cycles, double *starts, double *finishes,
                         double *service_out, long long *first_take_out) {
    Sim s;
    Fwd p;
    memset(&p, 0, sizeof(p));
    p.sms = (int)ints[0]; p.resident = (int)ints[1]; p.ncta = (int)ints[2]; p.warps = (int)ints[3];
    p.m = (int)ints[4]; p.n = (int)ints[5]; p.d = (int)ints[6]; p.q_len = (int)ints[7]; p.kv_len = (int)ints[8];
    p.dao = (int)ints[9]; p.output_element_bytes = (int)ints[10]; p.has_ranges = (int)ints[11];
    p.smem_rate = reals[0]; p.l2_rate = reals[1]; p.hbm_rate = reals[2]; p.tensor_rate = reals[3];
    p.vector_rate = reals[4]; p.sfu_rate = reals[5]; p.mma_latency = reals[6]; p.smem_latency = reals[7];
    p.l2_latency = reals[8]; p.hbm_latency = reals[9]; p.hbm_fraction = reals[10];

    int ff_on = (int)ints[12] && !p.has_ranges && p.resident <= 2, ff_n = (int)ints[13];
    double ff_rtol = reals[11];
    p.tiles = tiles; p.ranges = ranges; p.starts = starts; p.finishes = finishes;
    for (int i = 0; i < p.ncta; i++) { starts[i] = 0.0; finishes[i] = 0.0; }
    if (sim_init(&s, p.sms)) { sim_free(&s); return -1; }
    Callback cbs[8] = {0};
    cbs[CB_FWD_NEXT] = cb_fwd_next;
    cbs[CB_FWD_EPILOGUE] = cb_fwd_epilogue;
    cbs[CB_FWD_COMPLETE] = cb_fwd_complete;
    s.callbacks = cbs;
    s.program = &p;
    if (ff_on ? ffw_init(&s, p.sms, p.resident, p.ncta, ff_n, ff_rtol)
              : jump_init(&s, JUMP_FWD, p.sms, p.resident)) { sim_free(&s); return -1; }
    if ((int)ints[12] && !ff_on) ff_stats[0] = ff_stats[1] = 0;
    for (int slot = 0; slot < p.resident; slot++)
        for (int sm = 0; sm < p.sms; sm++) fwd_launch(&s, sm, slot);
    *out_cycles = run(&s);
    int rc = sim_output(&s, service_out, first_take_out);
    sim_free(&s);
    return rc;
}

typedef struct {
    int sms, resident, ncta, kind;
    long long m, n, d, a, b, q_len, kv_len, grid0, nq, batch, q_heads;
    int dao;
    double smem_rate, l2_rate, hbm_rate, tensor_rate, vector_rate, sfu_rate;
    double mma_latency, smem_latency, l2_latency, hbm_latency, fraction, atomic_fraction;
    const long long *minima;
    double *starts, *finishes;
    int next_cta;
} Bwd;

typedef struct {
    int cid, sm, slot;
    long long ntiles, remaining;
    long long lo, valid_n, lse_offset;
    int k, v, kvzero, qzero;
    int *atoms;
    long long natoms, catoms;
} BwdCta;

typedef struct {
    BwdCta *cta;
    long long mb;
    int next_q, next_do, next_lse;
} MainNext;

static void bwd_launch(Sim *s, int sm, int slot);

static int b_node(Sim *s, BwdCta *c, const Deps *d, int resource, double service, double latency, int cb,
                  void *ctx) {
    return add(s, d->v, d->n, resource, service, latency, c->cid, cb, ctx);
}

static int b_shared(Sim *s, BwdCta *c, double count, const Deps *d) {
    const Bwd *p = s->program;
    return b_node(s, c, d, R_SMEM(c->sm), count / p->smem_rate, count ? p->smem_latency : 0, 0, NULL);
}

static int b_read(Sim *s, BwdCta *c, double count, const Deps *d, int staging) {
    const Bwd *p = s->program;
    double l2_count = count;
    double hbm_count = count * p->fraction;
    int l2 = b_node(s, c, d, R_L2, l2_count / p->l2_rate, l2_count ? p->l2_latency : 0, 0, NULL);
    int hbm = b_node(s, c, d, R_HBM, hbm_count / p->hbm_rate, hbm_count ? p->hbm_latency : 0, 0, NULL);
    Deps x; D0(x); dpush(&x, l2); dpush(&x, hbm);
    return staging ? b_shared(s, c, count, &x) : b_node(s, c, &x, R_NONE, 0.0, 0.0, 0, NULL);
}

static int b_write(Sim *s, BwdCta *c, double count, const Deps *d, int atomic) {
    const Bwd *p = s->program;
    double l2_count = count;
    int l2 = b_node(s, c, d, R_L2, l2_count / p->l2_rate, l2_count ? p->l2_latency : 0, 0, NULL);
    double hbm_count = atomic ? count * (1 + p->atomic_fraction) : count * 1.0;
    int hbm = b_node(s, c, d, R_HBM, hbm_count / p->hbm_rate, hbm_count ? p->hbm_latency : 0, 0, NULL);
    Deps x; D0(x); dpush(&x, l2); dpush(&x, hbm);
    return b_node(s, c, &x, R_NONE, 0.0, 0.0, 0, NULL);
}

static int b_vector(Sim *s, BwdCta *c, double count, int warp, const Deps *d) {
    const Bwd *p = s->program;
    return b_node(s, c, d, R_VECTOR(c->sm, warp % 4), count / p->vector_rate, 0.0, 0, NULL);
}

static int b_sfu(Sim *s, BwdCta *c, double count, int warp, const Deps *d) {
    const Bwd *p = s->program;
    return b_node(s, c, d, R_SFU(c->sm, warp % 4), count / p->sfu_rate, 0.0, 0, NULL);
}

static int b_gemm(Sim *s, BwdCta *c, long long x, long long y, long long k, long long a, int warp,
                  const Deps *deps) {
    const Bwd *p = s->program;
    long long b = 8 / a;
    long long load_count = 2 * (x / a + y / b) * 16;
    double math_service = (double)(2 * (x / a) * (y / b) * 16) / p->tensor_rate;
    int tensor_resource = R_TENSOR(c->sm, warp % 4);
    double load_service = (double)load_count / p->smem_rate;
    double load_latency = load_count ? p->smem_latency : 0;
    double math_latency = p->mma_latency;
    int smem_resource = R_SMEM(c->sm);
    int last_load = -1, last_math = -1, prev_math = -1;
    for (long long group = 0; group < k / 16; group++) {
        Deps pre; dcopy(&pre, deps);
        if (last_load >= 0) dpush(&pre, last_load);
        if (group > 1) dpush(&pre, prev_math);
        int ld = add(s, pre.v, pre.n, smem_resource, load_service, load_latency, c->cid, 0, NULL);
        int md[2]; int nm = 0;
        md[nm++] = ld;
        if (last_math >= 0) md[nm++] = last_math;
        int mm = add(s, md, nm, tensor_resource, math_service, math_latency, c->cid, 0, NULL);
        last_load = ld;
        prev_math = last_math;
        last_math = mm;
    }
    return last_math;
}

static long long scalar_row_read_bytes(long long rows, long long warp_copies, long long row_offset) {
    long long groups = rows / 8, tail = rows % 8;
    long long offset = row_offset % 8;
    long long sectors = groups * (1 + (offset != 0));
    if (tail) sectors += (offset + tail + 7) / 8;
    return sectors * 32 * warp_copies;
}

static void bwd_complete(Sim *s, BwdCta *c, int done) {

    (void)done;
    Bwd *p = (Bwd *)s->program;
    p->finishes[c->cid] = s->now;
    int sm = c->sm, slot = c->slot;
    if (s->slots) s->slots[slot * p->sms + sm] = NULL;
    release(s, c->k); release(s, c->v); release(s, c->kvzero); release(s, c->qzero);
    for (long long i = 0; i < c->natoms; i++) release(s, c->atoms[i]);
    free(c->atoms);
    free(c);
    bwd_launch(s, sm, slot);
}

static void cb_bwd_finish(Sim *s, int id) { bwd_complete(s, s->nodes[id].ctx, id); }

static void simple_program(Sim *s, BwdCta *c) {
    const Bwd *p = s->program;
    long long m = p->m, d = p->d;
    int convert = p->kind == 1;
    long long vm = p->q_len - (c->cid % p->grid0) * m;
    long long valid_m = m < vm ? m : vm;
    Deps none; D0(none);
    int incoming = b_read(s, c, (double)(convert ? m * d * 4 : valid_m * d * 4), &none, 0);
    long long ops = convert ? 2 * m * d : 16 * m + m * (d - 8) + 24 * m;
    Deps work; D0(work);
    Deps in; D0(in); dpush(&in, incoming);
    for (int w = 0; w < 8; w++) dpush(&work, b_vector(s, c, (double)ops / 8.0, w, &in));
    if (convert) {
        int staged = b_shared(s, c, (double)(m * d * 2), &work);
        Deps st; D0(st); dpush(&st, staged);
        int one = b_shared(s, c, (double)(m * d * 2), &st);
        D0(work); dpush(&work, one);
    }
    long long outgoing = convert ? valid_m * d * 2 : m * (d * 4 + 4);
    int out = b_write(s, c, (double)outgoing, &work, 0);
    Deps fin; D0(fin); dpush(&fin, out);
    b_node(s, c, &fin, R_NONE, 0.0, 0.0, CB_BWD_FINISH, c);
}

static void main_tile(Sim *s, BwdCta *c, long long mb, const Deps *previous, int ready_q, int ready_do,
                      int ready_lse);

static void main_epilogue(Sim *s, BwdCta *c, const Deps *deps) {
    const Bwd *p = s->program;
    long long n = p->n, d = p->d;
    Deps scale; D0(scale);
    for (int w = 0; w < 8; w++) dpush(&scale, b_vector(s, c, (double)(n * d) / 8.0, w, deps));
    int stage = b_shared(s, c, (double)(2 * n * d * 2), &scale);
    Deps x; D0(x); dpush(&x, stage);
    int loaded = b_shared(s, c, (double)(2 * n * d * 2), &x);
    D0(x); dpush(&x, loaded);
    int out = b_write(s, c, (double)(2 * c->valid_n * d * 2), &x, 0);

    int *v = malloc((size_t)(c->natoms + 1) * sizeof(int));
    if (!v) { s->failed = 1; return; }
    memcpy(v, c->atoms, (size_t)c->natoms * sizeof(int));
    v[c->natoms] = out;
    add(s, v, (int)(c->natoms + 1), R_NONE, 0.0, 0.0, c->cid, CB_BWD_FINISH, c);
    free(v);
}

static void cb_main_next(Sim *s, int id) {
    MainNext *next = s->nodes[id].ctx;
    Deps prev; D0(prev); dpush(&prev, id);
    main_tile(s, next->cta, next->mb, &prev, next->next_q, next->next_do, next->next_lse);
    release(s, next->next_q); release(s, next->next_do); release(s, next->next_lse);
    free(next);
}

static void cb_main_epilogue(Sim *s, int id) {
    Deps prev; D0(prev); dpush(&prev, id);
    main_epilogue(s, s->nodes[id].ctx, &prev);
}

static void main_tile(Sim *s, BwdCta *c, long long mb, const Deps *previous, int ready_q, int ready_do,
                      int ready_lse) {
    const Bwd *p = s->program;
    long long m = p->m, n = p->n, d = p->d, a = p->a, b = p->b;
    c->ntiles++;
    c->remaining = mb - c->lo;
    if (s->jump && c->sm == 0 && c->slot == 0) s->snap_due = 1;
    long long ff_r; double ff_d;
    if (s->ff && ff_tile(s, (CtaHead *)c, &ff_r, &ff_d)) {

        Deps y = *previous;
        dpush(&y, ready_q); dpush(&y, ready_do); dpush(&y, ready_lse);
        int skip = b_node(s, c, &y, R_NONE, 0.0, ff_d, 0, NULL);
        MainNext *next = malloc(sizeof(MainNext));
        if (!next) { s->failed = 1; return; }
        next->cta = c; next->mb = mb - ff_r; next->next_q = skip; next->next_do = skip; next->next_lse = skip;
        hold(s, skip); hold(s, skip); hold(s, skip);
        Deps z; D0(z); dpush(&z, skip);
        b_node(s, c, &z, R_NONE, 0.0, 0.0, CB_MAIN_NEXT, next);
        return;
    }
    Deps x = *previous;
    dpush(&x, ready_q); dpush(&x, ready_do); dpush(&x, c->k); dpush(&x, c->v); dpush(&x, c->kvzero);
    dpush(&x, c->qzero);
    int begin = b_node(s, c, &x, R_NONE, 0.0, 0.0, 0, NULL);
    Deps bg; D0(bg); dpush(&bg, begin);
    int ds = b_read(s, c, (double)scalar_row_read_bytes(m, b, 0), &bg, 0);
    Deps s_ready, ds_done; D0(s_ready); D0(ds_done);
    for (int w = 0; w < 8; w++) {
        int qk = b_gemm(s, c, m, n, d, a, w, &bg);
        D0(x); dpush(&x, qk); dpush(&x, ready_lse);
        int score = b_vector(s, c, (double)((p->dao ? 2 : 1) * m * n) / 8.0 + (double)(4 * b * m) / 8.0, w, &x);
        D0(x); dpush(&x, score);
        int exp = b_sfu(s, c, (double)(m * n) / 8.0, w, &x);
        D0(x); dpush(&x, exp);
        int pstore = b_shared(s, c, (double)(m * n * 2) / 8.0, &x);
        D0(x); dpush(&x, pstore);
        int dp = b_gemm(s, c, m, n, d, a, w, &x);
        D0(x); dpush(&x, dp); dpush(&x, ds);
        int dswork = b_vector(s, c, (double)(2 * m * n) / 8.0, w, &x);
        dpush(&ds_done, dswork);
        D0(x); dpush(&x, dswork);
        dpush(&s_ready, b_shared(s, c, (double)(m * n * 2) / 8.0, &x));
    }
    int barrier = b_node(s, c, &s_ready, R_NONE, 0.0, 0.0, 0, NULL);
    int next_q = -1, next_do = -1;
    if (mb > c->lo) next_q = b_read(s, c, (double)(m * d * 2), &ds_done, 1);
    Deps dv; D0(dv);
    Deps br; D0(br); dpush(&br, barrier);
    for (int w = 0; w < 8; w++) dpush(&dv, b_gemm(s, c, n, d, m, 4, w, &br));
    int dvbar = b_node(s, c, &dv, R_NONE, 0.0, 0.0, 0, NULL);
    Deps dvb; D0(dvb); dpush(&dvb, dvbar);
    if (mb > c->lo) next_do = b_read(s, c, (double)(m * d * 2), &dvb, 1);
    Deps dk, dqs; D0(dk); D0(dqs);
    for (int w = 0; w < 8; w++) {
        int dq = b_gemm(s, c, m, d, n, a, w, &dvb);
        dpush(&dqs, dq);
        D0(x); dpush(&x, dq);
        int atom = b_write(s, c, (double)(m * d * 4) / 8.0, &x, 1);
        c->atoms = grow(c->atoms, &c->catoms, c->natoms + 1, sizeof(int), s);
        if (s->failed) return;
        c->atoms[c->natoms++] = atom;
        hold(s, atom);
        dpush(&dk, b_gemm(s, c, n, d, m, 4, w, &x));
    }
    if (mb > c->lo) {
        int next_lse = b_read(s, c, (double)scalar_row_read_bytes(m, b, c->lse_offset), &dqs, 0);
        MainNext *next = malloc(sizeof(MainNext));
        if (!next) { s->failed = 1; return; }
        next->cta = c; next->mb = mb - 1; next->next_q = next_q; next->next_do = next_do; next->next_lse = next_lse;
        hold(s, next_q); hold(s, next_do); hold(s, next_lse);
        b_node(s, c, &dk, R_NONE, 0.0, 0.0, CB_MAIN_NEXT, next);
    } else {
        b_node(s, c, &dk, R_NONE, 0.0, 0.0, CB_MAIN_EPILOGUE, c);
    }
}

static void main_program(Sim *s, BwdCta *c) {
    const Bwd *p = s->program;
    long long m = p->m, n = p->n, d = p->d, b = p->b;
    long long nb = c->cid % p->grid0;
    c->lo = p->minima[nb];
    long long vn = p->kv_len - nb * n;
    c->valid_n = n < vn ? n : vn;
    Deps none; D0(none);
    c->k = b_read(s, c, (double)(c->valid_n * d * 2), &none, 1); hold(s, c->k);
    c->v = b_read(s, c, (double)(c->valid_n * d * 2), &none, 1); hold(s, c->v);
    c->kvzero = b_shared(s, c, (double)(2 * (n - c->valid_n) * d * 2), &none); hold(s, c->kvzero);
    long long fr = p->q_len - (p->nq - 1) * m;
    long long first_rows = m < fr ? m : fr;
    int q = b_read(s, c, (double)(first_rows * d * 2), &none, 1);
    int dO = b_read(s, c, (double)(first_rows * d * 2), &none, 1);
    c->qzero = b_shared(s, c, (double)(2 * (m - first_rows) * d * 2), &none); hold(s, c->qzero);
    long long bidb = (c->cid / p->grid0) % p->batch;
    long long bidh = c->cid / (p->grid0 * p->batch);
    c->lse_offset = (bidb * p->q_heads + bidh) * p->q_len;
    int first_lse = b_read(s, c, (double)scalar_row_read_bytes(first_rows, b, c->lse_offset), &none, 0);
    main_tile(s, c, p->nq - 1, &none, q, dO, first_lse);
}

static void bwd_launch(Sim *s, int sm, int slot) {
    Bwd *p = (Bwd *)s->program;
    if (p->next_cta == p->ncta) return;
    int cid = p->next_cta++;
    p->starts[cid] = s->now;
    BwdCta *c = calloc(1, sizeof(BwdCta));
    if (!c) { s->failed = 1; return; }
    c->cid = cid; c->sm = sm; c->slot = slot;
    if (s->slots) s->slots[slot * p->sms + sm] = c;
    if (s->ff) ff_launch(s, (CtaHead *)c);
    c->k = c->v = c->kvzero = c->qzero = -1;
    if (p->kind == 2) main_program(s, c); else simple_program(s, c);
}

int flash_native_backward(const long long *ints, const double *reals, const long long *minima,
                          double *out_cycles, double *starts, double *finishes,
                          double *service_out, long long *first_take_out) {
    Sim s;
    Bwd p;
    memset(&p, 0, sizeof(p));
    p.sms = (int)ints[0]; p.resident = (int)ints[1]; p.ncta = (int)ints[2]; p.kind = (int)ints[3];
    p.m = ints[4]; p.n = ints[5]; p.d = ints[6]; p.a = ints[7]; p.b = ints[8]; p.q_len = ints[9];
    p.kv_len = ints[10]; p.grid0 = ints[11]; p.nq = ints[12]; p.batch = ints[13]; p.q_heads = ints[14];
    p.dao = (int)ints[15];
    p.smem_rate = reals[0]; p.l2_rate = reals[1]; p.hbm_rate = reals[2]; p.tensor_rate = reals[3];
    p.vector_rate = reals[4]; p.sfu_rate = reals[5]; p.mma_latency = reals[6]; p.smem_latency = reals[7];
    p.l2_latency = reals[8]; p.hbm_latency = reals[9]; p.fraction = reals[10]; p.atomic_fraction = reals[11];
    int ff_on = (int)ints[16] && p.kind == 2, ff_n = (int)ints[17];
    double ff_rtol = reals[12];
    p.minima = minima; p.starts = starts; p.finishes = finishes;
    for (int i = 0; i < p.ncta; i++) { starts[i] = 0.0; finishes[i] = 0.0; }
    if (sim_init(&s, p.sms)) { sim_free(&s); return -1; }
    Callback cbs[8] = {0};
    cbs[CB_BWD_FINISH] = cb_bwd_finish;
    cbs[CB_MAIN_NEXT] = cb_main_next;
    cbs[CB_MAIN_EPILOGUE] = cb_main_epilogue;
    s.callbacks = cbs;
    s.program = &p;
    if (ff_on ? ff_init(&s, p.sms, p.resident, p.ncta, ff_n, ff_rtol)
              : (p.kind == 2 && jump_init(&s, JUMP_BWD, p.sms, p.resident))) { sim_free(&s); return -1; }
    for (int slot = 0; slot < p.resident; slot++)
        for (int sm = 0; sm < p.sms; sm++) bwd_launch(&s, sm, slot);
    *out_cycles = run(&s);
    int rc = sim_output(&s, service_out, first_take_out);
    sim_free(&s);
    return rc;
}

#define JMARGIN 1
#define JMIN_WINDOW 3
#define JVALS 1024
#define JWORD_NONE 0xfffffffffffffff0ULL
#define JWORD_DONE 0xfffffffffffffff1ULL
#define JWORD_FREE 0xfffffffffffffff2ULL
#define JWORD_EMPTY 0xfffffffffffffff3ULL

static int jump_enabled = 1;
void flash_native_test_set_jump(int on) { jump_enabled = on; }
static __thread long long jump_stats[4];
void flash_native_jump_stats(long long *out) { for (int i = 0; i < 4; i++) out[i] = jump_stats[i]; }

typedef struct { unsigned long long *w; long long n, cap; } Words;

#define JRING 4

typedef struct Jump {
    Words snap[JRING];
    double snap_now[JRING];
    long long *snap_ntiles[JRING];
    double *snap_sum[JRING];
    int valid[JRING];
    int cur, nheld, overflow;
    double tmin, tmax;
    int *canon;
    unsigned *stamp;
    long long ncanon_cap;
    unsigned stampno;
    long long ccanon;
    int *bfs;
    long long nbfs, cbfs;
    unsigned long long vals[JVALS];
    unsigned char used[JVALS];
    Event *tmp;
    long long ctmp;
    long long probe_after;
    long long backoff;
    int failures;
    int sms;
} Jump;

static unsigned long long dbits(double x) { unsigned long long u; memcpy(&u, &x, sizeof u); return u; }
static double bitsd(unsigned long long u) { double x; memcpy(&x, &u, sizeof x); return x; }

static void jump_note_values(Sim *s, double service, double duration) {
    Jump *J = s->jump;
    double v[2] = {service, duration};
    for (int i = 0; i < 2; i++) {
        unsigned long long b = dbits(v[i]);
        unsigned h = (unsigned)((b * 0x9E3779B97F4A7C15ULL) >> 54) & (JVALS - 1);
        for (int probe = 0; ; probe++) {
            if (probe == JVALS) { J->overflow = 1; return; }
            unsigned j = (h + (unsigned)probe) & (JVALS - 1);
            if (!J->used[j]) { J->used[j] = 1; J->vals[j] = b; break; }
            if (J->vals[j] == b) break;
        }
    }
}

static int is_tie(unsigned long long d, int eu) {
    if (d == 0) return 0;
    int E = (int)((d >> 52) & 0x7ff);
    if (E == 0 || E == 0x7ff || (d >> 63)) return 1;
    unsigned long long M = (d & ((1ULL << 52) - 1)) | (1ULL << 52);
    int z = __builtin_ctzll(M);
    return E - 1075 + z == eu - 1;
}

static int no_ties(Jump *J, int eu) {
    for (int j = 0; j < JVALS; j++) if (J->used[j] && is_tie(J->vals[j], eu)) return 0;
    return 1;
}

static int jump_init(Sim *s, int kind, int sms, int resident) {
    for (int i = 0; i < 4; i++) jump_stats[i] = 0;
    if (!jump_enabled) return 0;
    Jump *J = calloc(1, sizeof(Jump));
    s->nslots = sms * resident;
    s->slots = calloc((size_t)s->nslots, sizeof(void *));
    if (!J || !s->slots) { free(J); return -1; }
    for (int i = 0; i < JRING; i++) {
        J->snap_ntiles[i] = calloc((size_t)s->nslots, sizeof(long long));
        J->snap_sum[i] = calloc((size_t)s->nres, sizeof(double));
        if (!J->snap_ntiles[i] || !J->snap_sum[i]) return -1;
    }
    J->sms = sms;
    s->jump = J;
    s->jump_kind = kind;
    return 0;
}

static void jump_free(Sim *s) {
    Jump *J = s->jump;
    free(s->slots);
    s->slots = NULL;
    if (!J) return;
    for (int i = 0; i < JRING; i++) { free(J->snap[i].w); free(J->snap_ntiles[i]); free(J->snap_sum[i]); }
    free(J->canon); free(J->stamp); free(J->bfs); free(J->tmp);
    free(J);
    s->jump = NULL;
}

static __thread Words *JB;
static __thread Sim *JS;
static inline void W(unsigned long long x) {
    if (JB->n == JB->cap) {
        long long c = JB->cap ? 2 * JB->cap : 4096;
        unsigned long long *p = realloc(JB->w, (size_t)c * sizeof(unsigned long long));
        if (!p) { JS->failed = 1; return; }
        JB->w = p; JB->cap = c;
    }
    JB->w[JB->n++] = x;
}

static inline void note(Jump *J, double t) { if (t < J->tmin) J->tmin = t; if (t > J->tmax) J->tmax = t; }

static long long canon(Sim *s, int id) {
    Jump *J = s->jump;
    if (J->stamp[id] != J->stampno) {
        J->stamp[id] = J->stampno;
        J->canon[id] = (int)J->ccanon++;
        if (J->nbfs == J->cbfs) J->bfs = grow(J->bfs, &J->cbfs, J->nbfs + 1, sizeof(int), s);
        if (!s->failed) J->bfs[J->nbfs++] = id;
    }
    return J->canon[id];
}

static void cod(Sim *s, int id) {
    if (id < 0) W(JWORD_NONE);
    else if (s->nodes[id].is_done) W(JWORD_DONE);
    else W((unsigned long long)canon(s, id));
}

static int ev_cmp(const void *a, const void *b) {
    if (event_lt(JS, a, b)) return -1;
    if (event_lt(JS, b, a)) return 1;
    return 0;
}

static int order_cmp(const void *a, const void *b) {
    const OrderItem *x = a, *y = b;
    return order_lt(x, y) ? -1 : order_lt(y, x) ? 1 : 0;
}

static void waiter_record(Sim *s, int w) {
    Jump *J = s->jump;
    Waiter *x = &s->waiters[w];
    Waiters *ws = &s->waiting[x->res];
    note(J, x->arrival_time);
    W((unsigned long long)x->res);
    W((unsigned long long)(ws->arrivals - x->number));
    W(dbits(x->arrival_time - s->now));
    W((unsigned long long)(s->serial - x->arrival_serial));
    W((unsigned long long)(ws->nacq - x->first));
    W((unsigned long long)(ws->nacq - x->end));
    W((unsigned long long)x->queued);
    W((unsigned long long)canon(s, x->node));
}

static void snapshot(Sim *s) {
    Jump *J = s->jump;
    double now = s->now;
    JS = s;
    JB = &J->snap[J->cur];
    JB->n = 0;
    J->tmin = now; J->tmax = now;
    if (J->ncanon_cap < s->cnodes) {
        long long c = s->cnodes;
        int *cn = realloc(J->canon, (size_t)c * sizeof(int));
        unsigned *st = realloc(J->stamp, (size_t)c * sizeof(unsigned));
        if (!cn || !st) { s->failed = 1; return; }
        memset(st + J->ncanon_cap, 0, (size_t)(c - J->ncanon_cap) * sizeof(unsigned));
        J->canon = cn; J->stamp = st; J->ncanon_cap = c;
    }
    J->stampno++;
    J->ccanon = 0;
    J->nbfs = 0;

    long long ne = s->now_queue.n + s->later.n;
    if (J->ctmp < ne) {
        J->tmp = grow(J->tmp, &J->ctmp, ne, sizeof(Event), s);
        if (s->failed) return;
    }
    memcpy(J->tmp, s->now_queue.h, (size_t)s->now_queue.n * sizeof(Event));
    memcpy(J->tmp + s->now_queue.n, s->later.h, (size_t)s->later.n * sizeof(Event));
    qsort(J->tmp, (size_t)ne, sizeof(Event), ev_cmp);
    W((unsigned long long)ne);
    for (long long i = 0; i < ne; i++) {
        Event *e = &J->tmp[i];
        note(J, e->t);
        W(dbits(e->t - now));
        W((unsigned long long)e->hi);
        if (e->waiter >= 0) { W(1); waiter_record(s, e->waiter); }
        else { W(0); W((unsigned long long)(s->serial - (long long)e->serial)); }
        W((unsigned long long)canon(s, e->node));
    }

    for (int r = 0; r < s->nres; r++) {
        Waiters *ws = &s->waiting[r];
        if (!ws->exists || !ws->norder) continue;
        W((unsigned long long)r);
        W((unsigned long long)ws->norder);
        OrderItem *items = malloc((size_t)ws->norder * sizeof(OrderItem));
        if (!items) { s->failed = 1; return; }
        memcpy(items, ws->order, (size_t)ws->norder * sizeof(OrderItem));
        qsort(items, (size_t)ws->norder, sizeof(OrderItem), order_cmp);
        long long first = ws->nacq;
        for (int i = 0; i < ws->norder; i++) {
            W((unsigned long long)items[i].owner);
            waiter_record(s, items[i].waiter);
            if (s->waiters[items[i].waiter].first < first) first = s->waiters[items[i].waiter].first;
        }
        free(items);
        W((unsigned long long)(ws->nacq - first));
        for (long long j = first; j < ws->nacq; j++) { note(J, ws->acquired[j]); W(dbits(ws->acquired[j] - now)); }
    }

    for (int r = 0; r < s->nres; r++) {
        if (s->busy[r] > now) { note(J, s->busy[r]); W(dbits(s->busy[r] - now)); }
        else W(JWORD_FREE);
    }

    for (int i = 0; i < s->nslots; i++) {
        CtaHead *h = s->slots[i];
        if (!h) { W(JWORD_EMPTY); continue; }
        W((unsigned long long)h->cid);
        if (s->jump_kind == JUMP_BWD) {
            BwdCta *c = (BwdCta *)h;
            long long k = 0;
            for (long long j = 0; j < c->natoms; j++) {
                int a = c->atoms[j];
                if (s->nodes[a].is_done) release(s, a);
                else c->atoms[k++] = a;
            }
            c->natoms = k;
            cod(s, c->k); cod(s, c->v); cod(s, c->kvzero); cod(s, c->qzero);
            W((unsigned long long)c->natoms);
            for (long long j = 0; j < c->natoms; j++) W((unsigned long long)canon(s, c->atoms[j]));
        } else {
            cod(s, ((FwdCta *)h)->qload);
        }
    }

    for (long long b = 0; b < J->nbfs; b++) {
        int id = J->bfs[b];
        Node *nd = &s->nodes[id];
        W((unsigned long long)(long long)nd->resource);
        W(dbits(nd->service));
        W(dbits(nd->duration));
        W((unsigned long long)nd->owner);
        W((unsigned long long)nd->pending);
        W((unsigned long long)nd->is_done);
        if (!nd->is_done) { note(J, nd->ready); W(dbits(nd->ready - now)); }
        W((unsigned long long)nd->refs);
        W((unsigned long long)nd->cb);
        if (nd->cb == CB_MAIN_NEXT) {
            MainNext *m = nd->ctx;
            W((unsigned long long)m->cta->cid);
            W((unsigned long long)(m->mb + m->cta->ntiles));
            cod(s, m->next_q); cod(s, m->next_do); cod(s, m->next_lse);
        } else if (nd->cb == CB_FWD_NEXT) {
            FwdNext *f = nd->ctx;
            W((unsigned long long)f->cta->cid);
            W((unsigned long long)(f->index - f->cta->ntiles));
            cod(s, f->next_k);
        } else if (nd->cb) {
            W((unsigned long long)((CtaHead *)nd->ctx)->cid);
        }
        nd = &s->nodes[id];
        W((unsigned long long)nd->nchild);
        for (int c = 0; c < nd->nchild; c++) W((unsigned long long)canon(s, s->nodes[id].children[c]));
    }
}

static int binade(double x, double *lo, double *hi, int *eu) {
    unsigned long long b = dbits(x);
    int E = (int)((b >> 52) & 0x7ff);
    if (x <= 0 || E == 0 || E >= 0x7fe) return 0;
    *lo = bitsd((unsigned long long)E << 52);
    *hi = bitsd((unsigned long long)(E + 1) << 52);
    *eu = E - 1075;
    return 1;
}

static void jump_point(Sim *s) {
    s->snap_due = 0;
    Jump *J = s->jump;
    if (!J || J->overflow || s->failed) return;
    if (!s->slots[0]) { J->nheld = 0; return; }
    if (J->probe_after > 0) { J->probe_after--; return; }
    for (int i = 0; i < s->nslots; i++) {
        CtaHead *h = s->slots[i];
        if (!h || h->remaining < JMIN_WINDOW) { J->nheld = 0; return; }
    }
    snapshot(s);
    if (s->failed) return;
    jump_stats[0]++;
    double lo = 0, hi = 0; int eu = 0;
    int c = J->cur;
    int inside = binade(s->now, &lo, &hi, &eu) && J->tmin >= lo && J->tmax < hi;
    J->snap_now[c] = s->now;
    J->valid[c] = inside;
    for (int i = 0; i < s->nslots; i++) J->snap_ntiles[c][i] = ((CtaHead *)s->slots[i])->ntiles;
    memcpy(J->snap_sum[c], s->service, (size_t)s->nres * sizeof(double));

    int ref = -1;
    for (int p = 1; inside && p < JRING && p <= J->nheld; p++) {
        int r = (c - p + JRING) % JRING;
        if (J->valid[r] && J->snap_now[r] >= lo && J->snap[r].n == J->snap[c].n
                && !memcmp(J->snap[r].w, J->snap[c].w, (size_t)J->snap[c].n * sizeof(unsigned long long))) { ref = r; break; }
    }
    if (ref >= 0) {
        jump_stats[1]++;
        J->backoff = 0;
        J->failures = 0;
        double P = s->now - J->snap_now[ref];
        const double *prev_sum = J->snap_sum[ref];
        long long k = -1;
        long long *dc = malloc((size_t)s->nslots * sizeof(long long));
        if (!dc) { s->failed = 1; return; }
        for (int i = 0; i < s->nslots; i++) {
            CtaHead *h = s->slots[i];
            dc[i] = h->ntiles - J->snap_ntiles[ref][i];
            if (dc[i] > 0) {
                long long ki = (h->remaining - JMARGIN) / dc[i];
                if (k < 0 || ki < k) k = ki;
            }
        }
        if (k < 0) k = 0;
        double room = (hi - J->tmax) / P - 2.0;
        if (room < (double)k) k = room > 0 ? (long long)room : 0;
        if (!no_ties(J, eu)) k = 0;
        for (int r = 0; r < s->nres && k > 0; r++) {
            double d = s->service[r] - prev_sum[r];
            if (d == 0) continue;
            double rlo, rhi; int reu;
            if (!binade(s->service[r], &rlo, &rhi, &reu) || prev_sum[r] < rlo || !no_ties(J, reu)) { k = 0; break; }
            double rroom = (rhi - s->service[r]) / d - 1.0;
            if (rroom < (double)k) k = rroom > 0 ? (long long)rroom : 0;
        }
        if (k > 0) {
            double D = (double)k * P;
            for (long long i = 0; i < s->now_queue.n; i++) s->now_queue.h[i].t += D;
            for (long long i = 0; i < s->later.n; i++) s->later.h[i].t += D;
            for (int r = 0; r < s->nres; r++) if (s->busy[r] > s->now) s->busy[r] += D;
            for (int w = 0; w < s->nwaiters; w++) if (s->waiters[w].node >= 0) s->waiters[w].arrival_time += D;
            for (int r = 0; r < s->nres; r++) {
                Waiters *ws = &s->waiting[r];
                if (!ws->exists || !ws->norder) continue;
                long long first = ws->nacq;
                for (int i = 0; i < ws->norder; i++)
                    if (s->waiters[ws->order[i].waiter].first < first) first = s->waiters[ws->order[i].waiter].first;
                for (long long j = first; j < ws->nacq; j++) ws->acquired[j] += D;
            }
            for (int id = 0; id < s->nnodes; id++) {
                Node *nd = &s->nodes[id];
                if (nd->is_done) continue;
                nd->ready += D;
                if (nd->cb == CB_MAIN_NEXT) {
                    MainNext *m = nd->ctx;
                    m->mb -= k * dc[m->cta->slot * J->sms + m->cta->sm];
                } else if (nd->cb == CB_FWD_NEXT) {
                    FwdNext *f = nd->ctx;
                    f->index += k * dc[f->cta->slot * J->sms + f->cta->sm];
                }
            }
            for (int r = 0; r < s->nres; r++) {
                double d = s->service[r] - prev_sum[r];
                if (d != 0) s->service[r] += (double)k * d;
            }
            for (int i = 0; i < s->nslots; i++) {
                CtaHead *h = s->slots[i];
                h->ntiles += k * dc[i];
                h->remaining -= k * dc[i];
            }
            s->now += D;
            jump_stats[2]++;
            jump_stats[3] += k;
        }
        free(dc);
        J->nheld = 0;
        return;
    }
    if (J->nheld && ++J->failures >= 16) {
        J->backoff = J->backoff ? (J->backoff < 64 ? 2 * J->backoff : 64) : 1;
        J->probe_after = J->backoff;
        J->failures = 0;
        J->nheld = 0;
        return;
    }
    J->cur = (c + 1) % JRING;
    if (J->nheld < JRING - 1) J->nheld++;
}

#define FF_HMAX 64
#define FF_NRES 15
#define FF_TAIL 2

typedef struct FF {
    int n, w, sms, nslots, ncta;
    double rtol;
    int *slot_of;
    double *hist;
    double *cum;
    double *cumsnap;
    double *period;
    int *last_wave;
    long long *plan_k;
    double *plan_T;
    unsigned *plan_mask;
} FF;

void flash_native_ff_stats(long long *out) { out[0] = ff_stats[0]; out[1] = ff_stats[1]; }

static int ff_init(Sim *s, int sms, int resident, int ncta, int n, double rtol) {
    ff_stats[0] = ff_stats[1] = 0;
    FF *f = calloc(1, sizeof(FF));
    if (!f) return -1;
    s->ff = f;
    f->n = n; f->w = resident; f->rtol = rtol; f->sms = sms; f->nslots = sms * resident; f->ncta = ncta;
    if (n < 1 || resident > 32 || f->w * (n + 1) + 1 > FF_HMAX) return -1;
    if (!s->slots) {
        s->nslots = f->nslots;
        s->slots = calloc((size_t)s->nslots, sizeof(void *));
        if (!s->slots) return -1;
    }
    f->slot_of = malloc((size_t)(ncta > 0 ? ncta : 1) * sizeof(int));
    f->hist = calloc((size_t)f->nslots * FF_HMAX, sizeof(double));
    f->cum = calloc((size_t)f->nslots * FF_NRES, sizeof(double));
    f->cumsnap = calloc((size_t)f->nslots * FF_HMAX * FF_NRES, sizeof(double));
    f->period = calloc((size_t)f->nslots, sizeof(double));
    f->last_wave = calloc((size_t)f->nslots, sizeof(int));
    f->plan_k = calloc((size_t)sms, sizeof(long long));
    f->plan_T = calloc((size_t)sms, sizeof(double));
    f->plan_mask = calloc((size_t)sms, sizeof(unsigned));
    if (!f->slot_of || !f->hist || !f->cum || !f->cumsnap || !f->period || !f->last_wave || !f->plan_k
            || !f->plan_mask || !f->plan_T) return -1;
    for (int i = 0; i < ncta; i++) f->slot_of[i] = -1;
    return 0;
}

static void ff_free(Sim *s) {
    FF *f = s->ff;
    if (!f) return;
    if (!s->jump) { free(s->slots); s->slots = NULL; }
    free(f->slot_of); free(f->hist); free(f->cum); free(f->cumsnap); free(f->period); free(f->last_wave);
    free(f->plan_k); free(f->plan_T); free(f->plan_mask);
    free(f);
    s->ff = NULL;
}

static void ff_launch(Sim *s, CtaHead *h) {
    FF *f = s->ff;
    int slot = h->slot * f->sms + h->sm;
    f->slot_of[h->cid] = slot;
    memset(f->cum + (size_t)slot * FF_NRES, 0, FF_NRES * sizeof(double));
    f->period[slot] = 0.0;
    f->last_wave[slot] = h->cid >= f->ncta - f->nslots;
    f->plan_mask[h->sm] &= ~(1u << h->slot);
}

static void ff_take(Sim *s, int owner, int resource, double service) {
    FF *f = s->ff;
    if (owner < 0 || owner >= f->ncta) return;
    int slot = f->slot_of[owner];
    if (slot < 0) return;
    int local = resource < 2 ? 13 + resource : (resource - 2) % R_PER_SM;
    f->cum[(size_t)slot * FF_NRES + local] += service;
}

static void ff_apply(Sim *s, CtaHead *h, int slot, long long windows, long long t) {

    FF *f = s->ff;
    const double *now = f->cumsnap + ((size_t)slot * FF_HMAX + (size_t)(t % FF_HMAX)) * FF_NRES;
    const double *then = f->cumsnap + ((size_t)slot * FF_HMAX + (size_t)((t - f->w) % FF_HMAX)) * FF_NRES;
    for (int j = 0; j < FF_NRES; j++) {
        double win = now[j] - then[j];
        if (win <= 0) continue;
        int res = j >= 13 ? j - 13 : 2 + h->sm * R_PER_SM + j;
        s->service[res] += (double)windows * win;
        f->cum[(size_t)slot * FF_NRES + j] += (double)windows * win;
    }
    ff_stats[0]++;
    ff_stats[1] += windows * f->w;
}

static long long ff_multiple(double T, double W, double rtol) {
    if (W <= 0) return 0;
    long long q = (long long)(T / W + 0.5);
    if (q < 1) return 0;
    double e = T - (double)q * W;
    return (e < 0 ? -e : e) <= rtol * T ? q : 0;
}

static int ff_tile(Sim *s, CtaHead *h, long long *tiles, double *delay) {
    FF *f = s->ff;
    int slot = h->slot * f->sms + h->sm;
    long long t = h->ntiles - 1;
    f->hist[(size_t)slot * FF_HMAX + (size_t)(t % FF_HMAX)] = s->now;
    memcpy(f->cumsnap + ((size_t)slot * FF_HMAX + (size_t)(t % FF_HMAX)) * FF_NRES,
           f->cum + (size_t)slot * FF_NRES, FF_NRES * sizeof(double));
    unsigned bit = 1u << h->slot;
    if (f->plan_mask[h->sm] & bit) {
        f->plan_mask[h->sm] &= ~bit;
        long long k = f->plan_k[h->sm];
        double T = f->plan_T[h->sm];
        long long q = ff_multiple(T, f->period[slot], f->rtol);
        if (!q || t < f->w || h->remaining - k * q * f->w < FF_TAIL) return 0;
        ff_apply(s, h, slot, k * q, t);
        *tiles = k * q * f->w;
        *delay = (double)k * T;
        return 1;
    }
    f->period[slot] = 0.0;
    if (f->last_wave[slot] || t < (long long)f->w * f->n) return 0;
    const double *hs = f->hist + (size_t)slot * FF_HMAX;
    double W0 = hs[t % FF_HMAX] - hs[(t - f->w) % FF_HMAX];
    if (W0 <= 0) return 0;
    for (int j = 1; j < f->n; j++) {
        double Wj = hs[(t - (long long)j * f->w) % FF_HMAX] - hs[(t - (long long)(j + 1) * f->w) % FF_HMAX];
        if (fabs_(Wj - W0) > f->rtol * W0) return 0;
    }
    f->period[slot] = W0;
    if (f->plan_mask[h->sm]) return 0;

    double T = W0;
    for (int r = 0; r < f->w; r++) {
        int o = r * f->sms + h->sm;
        CtaHead *oh = s->slots[o];
        if (r == h->slot) continue;
        if (!oh || f->last_wave[o] || f->period[o] <= 0) return 0;
        if (f->period[o] > T) T = f->period[o];
    }
    long long q = ff_multiple(T, W0, f->rtol);
    if (!q) return 0;
    long long k = (h->remaining - FF_TAIL) / (q * f->w);
    unsigned others = 0;
    for (int r = 0; r < f->w; r++) {
        if (r == h->slot) continue;
        int o = r * f->sms + h->sm;
        CtaHead *oh = s->slots[o];
        long long qo = ff_multiple(T, f->period[o], f->rtol);
        if (!qo) return 0;
        long long ko = (oh->remaining - 1 - FF_TAIL) / (qo * f->w);
        if (ko < k) k = ko;
        others |= 1u << r;
    }
    if (k < 1) return 0;
    f->plan_k[h->sm] = k;
    f->plan_T[h->sm] = T;
    f->plan_mask[h->sm] = others;
    ff_apply(s, h, slot, k * q, t);
    *tiles = k * q * f->w;
    *delay = (double)k * T;
    return 1;
}

#define FW_H 64
#define FW_NRES 15

typedef struct FFW {
    int n, sms, nslots, ncta, resident;
    double rtol;
    int *cta_sm;
    double *hb;
    double *snap;
    double *cum;
    double *start;
    long long *valid;
    double class_w[2];
} FFW;

static int ffw_init(Sim *s, int sms, int resident, int ncta, int n, double rtol) {
    ff_stats[0] = ff_stats[1] = 0;
    FFW *f = calloc(1, sizeof(FFW));
    if (!f) return -1;
    s->ffw = f;
    f->n = n; f->rtol = rtol; f->sms = sms; f->nslots = sms * resident; f->ncta = ncta; f->resident = resident;
    if (n < 1 || resident > 2 || 2 * n + 2 > FW_H) return -1;
    if (!s->slots) {
        s->nslots = f->nslots;
        s->slots = calloc((size_t)s->nslots, sizeof(void *));
        if (!s->slots) return -1;
    }
    f->cta_sm = malloc((size_t)(ncta > 0 ? ncta : 1) * sizeof(int));
    f->hb = calloc((size_t)f->nslots * FW_H, sizeof(double));
    f->snap = calloc((size_t)f->nslots * FW_H * FW_NRES, sizeof(double));
    f->cum = calloc((size_t)sms * FW_NRES, sizeof(double));
    f->start = calloc((size_t)f->nslots, sizeof(double));
    f->valid = calloc((size_t)f->nslots, sizeof(long long));
    if (!f->cta_sm || !f->hb || !f->snap || !f->cum || !f->start || !f->valid) return -1;
    return 0;
}

static void ffw_free(Sim *s) {
    FFW *f = s->ffw;
    if (!f) return;
    if (!s->jump) { free(s->slots); s->slots = NULL; }
    free(f->cta_sm); free(f->hb); free(f->snap); free(f->cum); free(f->start); free(f->valid);
    free(f);
    s->ffw = NULL;
}

static void ffw_launch(Sim *s, FwdCta *c) {
    FFW *f = s->ffw;
    int slot = c->slot * f->sms + c->sm;
    f->cta_sm[c->cid] = c->sm;
    f->start[slot] = s->now;
    f->valid[slot] = 0;
}

static void ffw_take(Sim *s, int owner, int resource, double service) {
    FFW *f = s->ffw;
    if (owner < 0 || owner >= f->ncta) return;
    int local = resource < 2 ? 13 + resource : (resource - 2) % R_PER_SM;
    f->cum[(size_t)f->cta_sm[owner] * FW_NRES + local] += service;
}

static int ffw_steady(FFW *f, int slot, long long t, int per, int n, double *W) {
    if (t - (long long)per * n < f->valid[slot]) return 0;
    const double *h = f->hb + (size_t)slot * FW_H;
    double W0 = h[t % FW_H] - h[(t - per) % FW_H];
    if (W0 <= 0) return 0;
    for (int j = 1; j < n; j++) {
        double Wj = h[(t - (long long)j * per) % FW_H] - h[(t - (long long)(j + 1) * per) % FW_H];
        if (fabs_(Wj - W0) > f->rtol * W0) return 0;
    }
    *W = W0;
    return 1;
}

static int ffw_class(FFW *f, int cls, int slot, long long t, int per, double *W, int *n) {
    double cw = f->class_w[cls];
    if (cw > 0 && ffw_steady(f, slot, t, per, 1, W) && fabs_(*W - cw) <= f->rtol * cw) { *n = 1; return 1; }
    if (!ffw_steady(f, slot, t, per, f->n, W)) return 0;
    if (cw <= 0) f->class_w[cls] = *W;
    *n = f->n;
    return 1;
}

static void ffw_add(Sim *s, FFW *f, int slot, int sm, long long t, int per, long long times) {
    const double *now = f->snap + ((size_t)slot * FW_H + (size_t)(t % FW_H)) * FW_NRES;
    const double *then = f->snap + ((size_t)slot * FW_H + (size_t)((t - per) % FW_H)) * FW_NRES;
    for (int j = 0; j < FW_NRES; j++) {
        double win = now[j] - then[j];
        if (win <= 0) continue;
        int res = j >= 13 ? j - 13 : 2 + sm * R_PER_SM + j;
        s->service[res] += (double)times * win;
        f->cum[(size_t)sm * FW_NRES + j] += (double)times * win;
    }
}

static int ffw_bus_queued(Sim *s, int a, int b) {
    for (int r = R_L2; r <= R_HBM; r++) {
        Waiters *w = &s->waiting[r];
        if (!w->exists) continue;
        for (int i = 0; i < w->norder; i++) if (w->order[i].owner == a || w->order[i].owner == b) return 1;
    }
    return 0;
}

static void ffw_heapify(Sim *s, Heap *q) { for (long long i = q->n / 2 - 1; i >= 0; i--) ev_siftup(s, q, i); }

static void ffw_shift(Sim *s, int a, int b, int sm, double dt) {
    Heap *L = &s->later, *N = &s->now_queue;
    for (long long i = 0; i < L->n; i++) {
        int o = (int)(L->h[i].hi & 0x7fffffffu);
        if (o == a || o == b) L->h[i].t += dt;
    }
    long long j = 0;
    for (long long i = 0; i < N->n; i++) {
        Event e = N->h[i];
        int o = (int)(e.hi & 0x7fffffffu);
        if (o == a || o == b) {
            e.t += dt;
            if (L->n == L->cap) {
                L->h = grow(L->h, &L->cap, L->n + 1, sizeof(Event), s);
                if (s->failed) return;
            }
            L->h[L->n++] = e;
        } else {
            N->h[j++] = e;
        }
    }
    N->n = j;
    ffw_heapify(s, L);
    ffw_heapify(s, N);
    for (int k = 0; k < R_PER_SM; k++) {
        int r = 2 + sm * R_PER_SM + k;
        if (s->busy[r] > s->now) s->busy[r] += dt;
        Waiters *w = &s->waiting[r];
        if (!w->exists) continue;
        for (int i = 0; i < w->norder; i++) s->waiters[w->order[i].waiter].arrival_time += dt;
        for (long long i = 0; i < w->nacq; i++) w->acquired[i] += dt;
    }
}

static int ffw_tile(Sim *s, FwdCta *c, long long index, long long *tiles, double *delay) {
    FFW *f = s->ffw;
    int sm = c->sm, slot = c->slot * f->sms + sm;
    long long t = index;
    f->hb[(size_t)slot * FW_H + (size_t)(t % FW_H)] = s->now;
    memcpy(f->snap + ((size_t)slot * FW_H + (size_t)(t % FW_H)) * FW_NRES, f->cum + (size_t)sm * FW_NRES,
           FW_NRES * sizeof(double));
    int oslot = f->resident > 1 ? (1 - c->slot) * f->sms + sm : -1;
    FwdCta *y = oslot >= 0 ? s->slots[oslot] : NULL;
    double W;
    int n;
    if (!y) {

        if (!ffw_class(f, 0, slot, t, 1, &W, &n)) return 0;
        long long k = c->remaining - FF_TAIL;
        if (k < 1) return 0;
        ffw_add(s, f, slot, sm, t, 1, k);
        f->valid[slot] = t + k;
        ff_stats[0]++; ff_stats[1] += k;
        *tiles = k; *delay = (double)k * W;
        return 1;
    }
    if (y->cid < c->cid) return 0;
    if (!ffw_class(f, 1, slot, t, 2, &W, &n)) return 0;

    if (f->hb[(size_t)slot * FW_H + (size_t)((t - 2 * n) % FW_H)] < f->start[oslot]) return 0;
    long long u = y->cur;
    double Wy;
    if (!y->pend || !ffw_steady(f, oslot, u, 1, n, &Wy) || fabs_(Wy - W) > f->rtol * W) return 0;
    long long k = (c->remaining - FF_TAIL) / 2;
    if (y->remaining - FF_TAIL < k) k = y->remaining - FF_TAIL;
    if (k < 1) return 0;
    if (ffw_bus_queued(s, c->cid, y->cid)) return 0;
    double dt = (double)k * W;
    ffw_shift(s, c->cid, y->cid, sm, dt);
    y->pend->index += k;
    y->remaining -= k;
    y->ntiles += k;
    f->valid[oslot] = u + 1 + k;
    f->valid[slot] = t + 2 * k;
    ffw_add(s, f, slot, sm, t, 2, k);
    ff_stats[0]++; ff_stats[1] += 3 * k;
    *tiles = 2 * k; *delay = dt;
    return 1;
}
