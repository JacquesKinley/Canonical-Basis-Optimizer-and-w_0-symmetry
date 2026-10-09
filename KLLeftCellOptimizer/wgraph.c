#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <math.h>
// ==========================================
// CONFIGURATION
// ==========================================
// CHANGE THIS TO WHATEVER YOU WANT
#define PARTITION_LEN 2
const int PARTITION[2] = {3,2}; 

// Memory Constants
#define INITIAL_CAPACITY 600000 
#define MAX_CAPACITY 50000000 
#define HASH_SIZE 16777216 
#define HASH_MASK (HASH_SIZE - 1)

// ==========================================
// Data Structures (FLAT ARRAYS)
// ==========================================
typedef struct {
    int target;
    int weight;
} Edge;

typedef struct {
    Edge *edges;
    int size;
    int capacity;
} VertexFlat;

// --- Global Data ---
int N; 
int NUM_TABLEAUX = 0;
size_t CURRENT_CAPACITY = INITIAL_CAPACITY;

uint8_t *COL_INDS = NULL; 
uint8_t *ROW_INDS = NULL; 
uint32_t *DESCENTS = NULL; 
int32_t *TABLE = NULL; 
VertexFlat *FLAT_GRAPH = NULL; 

// --- Performance Cache ---
int8_t *WEIGHT_CACHE = NULL; 
int *CACHE_TAGS = NULL; 
int GLOBAL_TAG = 0;     

int *HASH_VALUES = NULL;
int *HASH_KEYS = NULL;

// ==========================================
// Memory Management
// ==========================================

void init_global_structs() {
    HASH_VALUES = malloc(HASH_SIZE * sizeof(int));
    HASH_KEYS = malloc(HASH_SIZE * sizeof(int));
    WEIGHT_CACHE = calloc(MAX_CAPACITY, sizeof(int8_t)); 
    CACHE_TAGS = calloc(MAX_CAPACITY, sizeof(int));
    
    if (!HASH_VALUES || !HASH_KEYS || !WEIGHT_CACHE || !CACHE_TAGS) {
        printf("Allocation failed.\n"); exit(1);
    }
    memset(HASH_VALUES, -1, HASH_SIZE * sizeof(int));
    memset(HASH_KEYS, -1, HASH_SIZE * sizeof(int));
}

void add_edge_flat(int u, int v, int w) {
    VertexFlat *vu = &FLAT_GRAPH[u];
    if (vu->size == vu->capacity) {
        int new_cap = (vu->capacity == 0) ? 8 : vu->capacity * 2;
        vu->edges = realloc(vu->edges, new_cap * sizeof(Edge));
        vu->capacity = new_cap;
    }
    vu->edges[vu->size].target = v;
    vu->edges[vu->size].weight = w;
    vu->size++;

    VertexFlat *vv = &FLAT_GRAPH[v];
    if (vv->size == vv->capacity) {
        int new_cap = (vv->capacity == 0) ? 8 : vv->capacity * 2;
        vv->edges = realloc(vv->edges, new_cap * sizeof(Edge));
        vv->capacity = new_cap;
    }
    vv->edges[vv->size].target = u;
    vv->edges[vv->size].weight = w;
    vv->size++;
}

int get_weight_flat(int u, int v) {
    VertexFlat *vu = &FLAT_GRAPH[u];
    for(int i=0; i<vu->size; i++) {
        if (vu->edges[i].target == v) return vu->edges[i].weight;
    }
    return 0;
}

// ==========================================
// Helpers
// ==========================================
uint32_t hash_seq(const uint8_t *data, int len) {
    uint32_t hash = 2166136261u;
    for (int i = 0; i < len; i++) {
        hash ^= data[i];
        hash *= 16777619;
    }
    return hash;
}

void hash_insert(int val) {
    uint8_t *key_data = &COL_INDS[val * N];
    uint32_t h = hash_seq(key_data, N) & HASH_MASK;
    while (HASH_VALUES[h] != -1) {
        h = (h + 1) & HASH_MASK;
    }
    HASH_KEYS[h] = val;
    HASH_VALUES[h] = val;
}

int hash_find(const uint8_t *target_seq) {
    uint32_t h = hash_seq(target_seq, N) & HASH_MASK;
    while (HASH_VALUES[h] != -1) {
        int idx = HASH_KEYS[h];
        if (memcmp(&COL_INDS[idx * N], target_seq, N) == 0) {
            return idx;
        }
        h = (h + 1) & HASH_MASK;
    }
    return -1;
}

void ensure_capacity() {
    if (NUM_TABLEAUX >= CURRENT_CAPACITY) {
        if (CURRENT_CAPACITY >= MAX_CAPACITY) {
            printf("[Error] Max capacity reached.\n"); exit(1);
        }
        size_t new_cap = CURRENT_CAPACITY * 2;
        
        uint8_t *new_col = realloc(COL_INDS, new_cap * N * sizeof(uint8_t));
        uint8_t *new_row = realloc(ROW_INDS, new_cap * N * sizeof(uint8_t));
        uint32_t *new_des = realloc(DESCENTS, new_cap * sizeof(uint32_t));
        int32_t *new_tbl = realloc(TABLE, new_cap * (N - 1) * sizeof(int32_t));
        
        if (!new_col || !new_row || !new_des || !new_tbl) {
            printf("OOM during resize.\n"); exit(1);
        }
        
        memset(new_tbl + CURRENT_CAPACITY * (N-1), 0, (new_cap - CURRENT_CAPACITY) * (N-1) * sizeof(int32_t));

        COL_INDS = new_col;
        ROW_INDS = new_row;
        DESCENTS = new_des;
        TABLE = new_tbl;
        CURRENT_CAPACITY = new_cap;
    }
}

// ==========================================
// Part 1: Tableau Generation
// ==========================================
void generate_tableaux() {
    N = 0;
    for(int i=0; i<PARTITION_LEN; i++) N += PARTITION[i];
    printf("Generating Tableaux (N=%d)...\n", N);

    COL_INDS = malloc(CURRENT_CAPACITY * N * sizeof(uint8_t));
    ROW_INDS = malloc(CURRENT_CAPACITY * N * sizeof(uint8_t));
    DESCENTS = malloc(CURRENT_CAPACITY * sizeof(uint32_t));
    TABLE = malloc(CURRENT_CAPACITY * (N - 1) * sizeof(int32_t));
    memset(TABLE, 0, CURRENT_CAPACITY * (N - 1) * sizeof(int32_t));
    
    // Initialize Structures based on N
    init_global_structs();

    int dual_pi[64] = {0};
    int max_col = PARTITION[0];
    for(int j=0; j<max_col; j++) {
        for(int i=0; i<PARTITION_LEN; i++) if(PARTITION[i] > j) dual_pi[j]++;
    }
    
    uint8_t r_seq[64], c_seq[64];
    int k = 0;
    for (int c = 0; c < max_col; c++) {
        for (int r = 0; r < dual_pi[c]; r++) {
            r_seq[k] = r + 1; c_seq[k] = c + 1; k++;
        }
    }
    
    memcpy(&COL_INDS[0], c_seq, N);
    memcpy(&ROW_INDS[0], r_seq, N);
    hash_insert(0);
    
    uint32_t desc = 0;
    for (int i = 0; i < N - 1; i++) if (r_seq[i] < r_seq[i+1]) desc |= (1U << i);
    DESCENTS[0] = desc;
    for (int i = 0; i < N - 1; i++) if ((desc >> i) & 1) TABLE[i] = -1;

    NUM_TABLEAUX = 1;
    int m = 0; int finished = 0;
    uint8_t r_curr[64], c_curr[64]; int pcl_arr[64];

    while (!finished) {
        memcpy(r_curr, &ROW_INDS[m * N], N);
        memcpy(c_curr, &COL_INDS[m * N], N);
        memset(pcl_arr, 0, sizeof(pcl_arr));
        int mag_j = 1; int pcl_len = 1; pcl_arr[1] = 1; 
        
        while (mag_j < N) {
            int c_next = c_curr[mag_j]; int r_next = r_curr[mag_j];
            int c_prev = c_curr[mag_j-1]; int r_prev = r_curr[mag_j-1];
            int cond = 0;
            if (c_next == 1) cond = 1;
            else if (c_next == 1 + c_prev && r_next == r_prev) cond = 1;
            if (!cond) break;
            if (r_next == 1) { pcl_len++; pcl_arr[pcl_len] = 1; } 
            else { pcl_arr[c_next]++; }
            mag_j++;
        }

        if (mag_j == N) { finished = 1; } 
        else {
            int c_next = c_curr[mag_j]; int r_next = r_curr[mag_j];
            if (r_next == 1) { pcl_len++; pcl_arr[pcl_len] = 1; } 
            else { pcl_arr[c_next]++; }
            int k_col = c_next - 1;
            while (k_col > 0 && pcl_arr[k_col] == r_next) k_col--;
            pcl_arr[k_col]--;

            uint8_t new_r[64], new_c[64]; int ptr = 0;
            for(int i=1; i<=pcl_len; i++) for(int h=1; h<=pcl_arr[i]; h++) { new_r[ptr] = h; new_c[ptr] = i; ptr++; }
            new_r[ptr] = pcl_arr[k_col] + 1; new_c[ptr] = k_col; ptr++;
            for(int x = mag_j + 1; x < N; x++) { new_r[ptr] = r_curr[x]; new_c[ptr] = c_curr[x]; ptr++; }
            
            ensure_capacity();
            int curr_idx = NUM_TABLEAUX;
            memcpy(&ROW_INDS[curr_idx * N], new_r, N);
            memcpy(&COL_INDS[curr_idx * N], new_c, N);
            hash_insert(curr_idx);
            NUM_TABLEAUX++;

            desc = 0;
            uint8_t temp_r[64], temp_c[64];
            for (int i = 0; i < N - 1; i++) {
                if (new_r[i] == new_r[i+1]) { TABLE[curr_idx * (N - 1) + i] = curr_idx + 1; }
                else if (new_c[i] == new_c[i+1]) { TABLE[curr_idx * (N - 1) + i] = -(curr_idx + 1); desc |= (1U << i); } 
                else if (new_r[i] < new_r[i+1]) { 
                    desc |= (1U << i);
                    memcpy(temp_r, new_r, N); memcpy(temp_c, new_c, N);
                    uint8_t tmp = temp_r[i]; temp_r[i] = temp_r[i+1]; temp_r[i+1] = tmp;
                    tmp = temp_c[i]; temp_c[i] = temp_c[i+1]; temp_c[i+1] = tmp;
                    int neighbor_ind = hash_find(temp_c);
                    if (neighbor_ind != -1) {
                        TABLE[curr_idx * (N - 1) + i] = neighbor_ind + 1; 
                        TABLE[neighbor_ind * (N - 1) + i] = curr_idx + 1;
                    }
                }
            }
            DESCENTS[curr_idx] = desc;
        }
        m++;
    }
    printf("Generated %d tableaux.\n", NUM_TABLEAUX);
    FLAT_GRAPH = calloc(NUM_TABLEAUX, sizeof(VertexFlat));
}

// ==========================================
// Part 2: Weighted Graph Logic
// ==========================================

// UNIVERSAL resNum (Works for any N)
static inline int resNum(int u_idx, int t_idx) {
    uint64_t *u64 = (uint64_t*)&COL_INDS[u_idx * N];
    uint64_t *t64 = (uint64_t*)&COL_INDS[t_idx * N];
    
    // We check 8 bytes at a time (uint64_t)
    // N bytes = (N+7)/8 blocks of 64-bit ints
    int num_blocks = (N + 7) / 8;
    
    for (int k = 0; k < num_blocks; k++) {
        uint64_t diff = u64[k] ^ t64[k];
        if (diff != 0) {
            int bit_idx = __builtin_ctzll(diff);
            int byte_offset = bit_idx >> 3; // divide by 8
            int result = (k * 8) + byte_offset;
            // Bound check: if padding caused difference, ignore (should not happen if zero-init)
            if (result < N) return result;
            else return 0;
        }
    }
    return 0;
}

int favut(int u_idx, int t_idx, int *fu, int *ft) {
    int r = resNum(u_idx, t_idx);
    if (r == 0) return 0;
    uint8_t *x = &COL_INDS[u_idx * N];
    uint8_t *y = &COL_INDS[t_idx * N];
    int v1 = x[r]; int v2 = y[r];
    int mn = (v1 < v2) ? v1 : v2;
    int mx = (v1 > v2) ? v1 : v2; mx--;
    int val_prev = x[r-1];
    if (val_prev >= mn && val_prev <= mx) {
        *fu = u_idx; *ft = t_idx; return 1;
    } else {
        int s = r - 2;
        while (s >= 0) { if (x[s] >= mn && x[s] <= mx) break; s--; }
        if (s < 0) return 0;
        uint8_t z[64]; memcpy(z, x, N);
        for (int k = s; k < r - 1; k++) z[k] = x[k+1];
        z[r-1] = x[s];
        uint8_t zy[64]; memcpy(zy, z, r); memcpy(zy + r, y + r, N - r); 
        *fu = hash_find(z); *ft = hash_find(zy);
        return (*fu != -1 && *ft != -1);
    }
}

void compute_weighted_graph() {
    printf("Computing Weighted Edges (Safe Optimized)...\n");
    long long edge_count = 0;
    
    // --- Progress Bar Setup ---
    int step = NUM_TABLEAUX / 100; // Update every 1%
    if (step == 0) step = 1;       // Safety for small runs
    
    // Phase 1 (Standard)
    for (int t = 0; t < NUM_TABLEAUX; t++) {
        uint32_t des_t = DESCENTS[t];
        for (int s = 0; s < N - 1; s++) {
            if ((des_t >> s) & 1) {
                int32_t val = TABLE[t * (N - 1) + s];
                if (val > 0) {
                    int u = val - 1; 
                    if (u < t) { add_edge_flat(u, t, 1); edge_count++; }
                }
            }
        }
    }
    printf("Phase 1 Edges: %lld\n", edge_count);
    printf("Phase 2 Progress:\n");

    // Phase 2: LINEAR LOOP (Safe Logic)
    for (int t = 1; t < NUM_TABLEAUX; t++) {
        
        // --- PROGRESS BAR UPDATE ---
        if (t % step == 0 || t == NUM_TABLEAUX - 1) {
            float percent = (float)t / (NUM_TABLEAUX - 1) * 100.0f;
            int bar_width = 40;
            int pos = (int)((float)bar_width * percent / 100.0f);
            
            printf("\r[");
            for (int i = 0; i < bar_width; ++i) {
                if (i < pos) printf("=");
                else if (i == pos) printf(">");
                else printf(" ");
            }
            printf("] %.1f%% (%d/%d)", percent, t, NUM_TABLEAUX);
            fflush(stdout); 
        }

        uint32_t des_t = DESCENTS[t];
        
        // Reverted to linear loop for correctness stability
        for (int u = 0; u < t; u++) {
             uint32_t des_u = DESCENTS[u];
             
             // Subset Check
             if ((des_t & des_u) == des_t && des_t != des_u) {
                int i = resNum(u, t);
                int j = -1;
                for (int s_idx = N - 2; s_idx >= 0; s_idx--) {
                    if ((des_t >> s_idx) & 1) {
                         if (TABLE[t * (N - 1) + s_idx] > 0) { j = s_idx + 1; break; }
                    }
                }

                if (j != -1 && i < j) {
                    int i_bit = i - 1; int j_bit = j - 1;
                    int in_diff = ((DESCENTS[u] >> i_bit) & 1) && !((des_t >> i_bit) & 1);
                    int fu = u, ft = t; 
                    if (!in_diff) { if (!favut(u, t, &fu, &ft)) { fu = u; ft = t; } }

                    uint32_t des_fu = DESCENTS[fu];
                    uint32_t des_ft = DESCENTS[ft];

                    int in_diff_f = ((des_fu >> i_bit) & 1) && !((des_ft >> i_bit) & 1);
                    
                    if (in_diff_f) {
                        int weight = 0;
                        if (ft < t) {
                            if (ft >= 0 && fu >= 0) {
                                 int w = get_weight_flat(ft, fu); 
                                 if (w != 0) weight = w;
                            }
                        } else {
                            // GENERATION ID CACHE
                            GLOBAL_TAG++;
                            
                            VertexFlat *v_fu = &FLAT_GRAPH[fu];
                            for(int z=0; z<v_fu->size; z++) {
                                int target = v_fu->edges[z].target;
                                WEIGHT_CACHE[target] = (int8_t)v_fu->edges[z].weight;
                                CACHE_TAGS[target] = GLOBAL_TAG;
                            }
                            WEIGHT_CACHE[fu] = 1; CACHE_TAGS[fu] = GLOBAL_TAG;

                            int32_t val_v = TABLE[ft * (N - 1) + j_bit];
                            int v = abs(val_v) - 1;
                            
                            if (v >= 0) {
                                uint32_t des_v = DESCENTS[v];
                                int i_in_des_v = (des_v >> i_bit) & 1;
                                
                                if (!i_in_des_v) {
                                     int sumji = 0, sumij = 0;
                                     VertexFlat *v_v = &FLAT_GRAPH[v];
                                     for(int z=0; z<v_v->size; z++) {
                                         int x = v_v->edges[z].target;
                                         int w_vx = v_v->edges[z].weight;
                                         int w_xfu = (CACHE_TAGS[x] == GLOBAL_TAG) ? WEIGHT_CACHE[x] : 0;
                                         
                                         if (w_xfu != 0) {
                                             uint32_t des_x = DESCENTS[x];
                                             if (x != ft && ((des_x >> j_bit)&1) && !((des_x >> i_bit)&1)) sumji += w_vx * w_xfu;
                                             if (((des_x >> i_bit)&1) && !((des_x >> j_bit)&1)) sumij += w_vx * w_xfu;
                                         }
                                     }
                                     weight = sumij - sumji;
                                }
                                else if (i == j - 1 && i_in_des_v) {
                                    int32_t val_w = TABLE[v * (N - 1) + i_bit];
                                    if (val_w > 0) {
                                        int w_idx = val_w - 1;
                                        int sumji = 0, sumij = 0;
                                        VertexFlat *v_w = &FLAT_GRAPH[w_idx];
                                        for(int z=0; z<v_w->size; z++) {
                                            int x = v_w->edges[z].target;
                                            int w_wx = v_w->edges[z].weight;
                                            uint32_t des_x = DESCENTS[x];
                                            int i_idx = (des_x >> i_bit) & 1;
                                            int j_idx = (des_x >> j_bit) & 1;
                                            
                                            int y = -1;
                                            if (x != v && i_idx && !j_idx) {
                                                int32_t t_jx = TABLE[x * (N - 1) + j_bit];
                                                if (abs(t_jx) - 1 > x) { 
                                                     int tx = abs(t_jx) - 1; if ( !((DESCENTS[tx] >> i_bit) & 1) ) y = tx;
                                                } else {
                                                     int32_t t_ix = TABLE[x * (N - 1) + i_bit];
                                                     if (abs(t_ix) - 1 < x && t_ix > 0) {
                                                          int tix = abs(t_ix) - 1; if ( (DESCENTS[tix] >> j_bit) & 1 ) y = tix;
                                                     }
                                                }
                                            }
                                            if (j_idx && !i_idx) {
                                                int32_t t_ix = TABLE[x * (N - 1) + i_bit];
                                                if (abs(t_ix) - 1 > x) {
                                                     int tix = abs(t_ix) - 1; if ( !((DESCENTS[tix] >> j_bit) & 1) ) y = tix;
                                                } else {
                                                     int32_t t_jx = TABLE[x * (N - 1) + j_bit];
                                                     if (abs(t_jx) - 1 < x && t_jx > 0) {
                                                          int tjx = abs(t_jx) - 1; if ( (DESCENTS[tjx] >> i_bit) & 1 ) y = tjx;
                                                     }
                                                }
                                            }
                                            
                                            if (y != -1) {
                                                int w_yfu = (CACHE_TAGS[y] == GLOBAL_TAG) ? WEIGHT_CACHE[y] : 0;
                                                if (w_yfu) sumji += w_wx * w_yfu;
                                            }
                                        }
                                        weight = sumji - sumij;
                                    }
                                }
                            }
                        }
                        
                        if (weight > 0) {
                            add_edge_flat(u, t, weight);
                            edge_count++;
                            if (weight != 1) printf("[!] Special Edge: %d -- %d (Weight: %d)\n", u+1, t+1, weight);
                        }
                    }
                }
             }
        }
    }
    printf("\nTotal Edges: %lld\n", edge_count);
}

void export_to_scipy_format(const char* filename) {
    printf("Exporting graph to %s...\n", filename);
    FILE *f = fopen(filename, "w");
    if (f == NULL) {
        printf("[Error] Could not open file for writing.\n");
        return;
    }

    // Iterate through every vertex (row)
    for (int row = 0; row < NUM_TABLEAUX; row++) {
        VertexFlat *v = &FLAT_GRAPH[row];
        
        // Iterate through every edge (column targets)
        for (int i = 0; i < v->size; i++) {
            int col = v->edges[i].target;
            int weight = v->edges[i].weight;
            
            // Write Triplet: Row Col Weight
            fprintf(f, "%d %d %d\n", row, col, weight);
        }
    }

    fclose(f);
    printf("Export complete.\n");
}

// Add this function above main()
void export_binary(const char* filename) {
    printf("Exporting BINARY graph to %s...\n", filename);
    FILE *f = fopen(filename, "wb"); // 'wb' is critical for binary
    if (!f) {
        printf("Error opening file!\n");
        return;
    }

    // Buffer: 3 integers (Row, Col, Weight)
    int buffer[3]; 

    for (int row = 0; row < NUM_TABLEAUX; row++) {
        VertexFlat *v = &FLAT_GRAPH[row];
        for (int i = 0; i < v->size; i++) {
            buffer[0] = row;                // Row Index
            buffer[1] = v->edges[i].target; // Col Index
            buffer[2] = v->edges[i].weight; // Weight
            
            // Write 12 bytes (3 * 4 bytes) directly to disk
            fwrite(buffer, sizeof(int), 3, f);
        }
    }
    fclose(f);
    printf("Binary export complete.\n");
}

// ==========================================
// Part 4: Compact Adjacency List Export
// ==========================================
void export_adjacency_binary(const char* filename) {
    printf("Exporting COMPACT Adjacency binary to %s...\n", filename);
    FILE *f = fopen(filename, "wb");
    if (!f) {
        printf("[Error] Could not open file for writing.\n");
        return;
    }

    // 1. Write Header: Total Number of Vertices (int32)
    int n = NUM_TABLEAUX;
    fwrite(&n, sizeof(int), 1, f);

    long long total_bytes = sizeof(int);

    // 2. Write Data for each vertex
    for (int i = 0; i < n; i++) {
        VertexFlat *v = &FLAT_GRAPH[i];
        int degree = v->size;
        
        // Write Degree of current vertex
        fwrite(&degree, sizeof(int), 1, f);
        total_bytes += sizeof(int);

        // Write Pairs: (Target, Weight)
        // We write them sequentially: T1, W1, T2, W2, ...
        for (int j = 0; j < degree; j++) {
            fwrite(&v->edges[j].target, sizeof(int), 1, f);
            fwrite(&v->edges[j].weight, sizeof(int), 1, f);
        }
        total_bytes += degree * 2 * sizeof(int);
    }

    fclose(f);
    printf("Export complete. Size: %.2f MB\n", (double)total_bytes / (1024*1024));
}

// ==========================================
// Part 3: Hecke Algebra Verification (TestWG)
// ==========================================

// Matrix-Vector Multiplication: out = T_s * in
// T_s is the Hecke generator for index s (0 <= s < N-1)
// q is the Hecke parameter
void apply_Ts(int s, double q, const double *in, double *out, int n_vertices) {
    double inv_q = 1.0 / q;
    
    // Initialize output with diagonal contributions
    for (int u = 0; u < n_vertices; u++) {
        // Diagonal: If s is in Descents(u), val is -q^-1, else q
        int s_in_des_u = (DESCENTS[u] >> s) & 1;
        double diag = s_in_des_u ? -inv_q : q;
        out[u] = diag * in[u];
    }

    // Add off-diagonal contributions
    // T_s[u][v] = weight(v->u) IF s in Des(u) AND s NOT in Des(v)
    // We iterate edges in FLAT_GRAPH. Since it stores undirected edges (u <-> v),
    // we must check the condition carefully.
    // The matrix action is y = M * x => y[u] = sum(M[u][v] * x[v])
    
    for (int u = 0; u < n_vertices; u++) {
        int s_in_des_u = (DESCENTS[u] >> s) & 1;
        
        // Optimization: If s is NOT in Des(u), the row u has no off-diagonal entries
        // (based on the definition: entry exists only if s in Label(u) and s not in Label(v))
        if (!s_in_des_u) continue;

        VertexFlat *vert = &FLAT_GRAPH[u];
        for (int i = 0; i < vert->size; i++) {
            int v = vert->edges[i].target;
            int w = vert->edges[i].weight;
            
            int s_in_des_v = (DESCENTS[v] >> s) & 1;

            if (!s_in_des_v) {
                out[u] += (double)w * in[v];
            }
        }
    }
}

// Helper: Check if two vectors are approximately equal
int check_vectors(const double *v1, const double *v2, int size, const char *test_name) {
    double max_diff = 0.0;
    for(int i=0; i<size; i++) {
        double diff = fabs(v1[i] - v2[i]);
        if(diff > max_diff) max_diff = diff;
    }
    
    if (max_diff > 1e-9) {
        printf("[FAIL] %s (Max Diff: %.2e)\n", test_name, max_diff);
        return 0;
    }
    return 1;
}

void test_hecke_relations() {
    printf("\nRunning Hecke Algebra Verification (q = 1.5)...\n");
    
    int n = NUM_TABLEAUX;
    if (n == 0) return;
    
    double q = 1.5; // Test value for q
    
    // Allocate vectors
    double *v0 = malloc(n * sizeof(double));
    double *v1 = malloc(n * sizeof(double));
    double *v2 = malloc(n * sizeof(double));
    double *v3 = malloc(n * sizeof(double)); // Temp
    double *v4 = malloc(n * sizeof(double)); // Temp

    // Initialize v0 with random values
    srand(time(NULL));
    for(int i=0; i<n; i++) v0[i] = ((double)rand() / RAND_MAX);

    int fails = 0;
    int tests = 0;

    // 1. Quadratic Relation: (T_s - q)(T_s + q^-1) = 0
    // Equivalent to checking: (T_s^2) v == ((q - q^-1)T_s + I) v
    printf("Checking Quadratic Relations...\n");
    for (int s = 0; s < N - 1; s++) {
        apply_Ts(s, q, v0, v1, n);        // v1 = T_s * v0
        apply_Ts(s, q, v1, v2, n);        // v2 = T_s * v1 = T_s^2 * v0
        
        double coef = q - (1.0/q);
        for(int i=0; i<n; i++) v3[i] = (coef * v1[i]) + v0[i]; // v3 = (q-q^-1)T_s v0 + v0
        
        char tag[64]; sprintf(tag, "Quad s=%d", s);
        if (!check_vectors(v2, v3, n, tag)) fails++;
        tests++;
    }

    // 2. Braid Relations: T_s T_{s+1} T_s = T_{s+1} T_s T_{s+1}
    printf("Checking Braid Relations...\n");
    for (int s = 0; s < N - 2; s++) {
        int t = s + 1;
        
        // LHS: s -> t -> s
        apply_Ts(s, q, v0, v1, n);
        apply_Ts(t, q, v1, v2, n);
        apply_Ts(s, q, v2, v3, n); // v3 = LHS
        
        // RHS: t -> s -> t
        apply_Ts(t, q, v0, v1, n);
        apply_Ts(s, q, v1, v2, n);
        apply_Ts(t, q, v2, v4, n); // v4 = RHS
        
        char tag[64]; sprintf(tag, "Braid s=%d", s);
        if (!check_vectors(v3, v4, n, tag)) fails++;
        tests++;
    }

    // 3. Commutation Relations: T_s T_t = T_t T_s if |s-t| >= 2
    printf("Checking Commutation Relations...\n");
    for (int s = 0; s < N - 1; s++) {
        for (int t = s + 2; t < N - 1; t++) {
            // LHS: s -> t
            apply_Ts(s, q, v0, v1, n);
            apply_Ts(t, q, v1, v3, n);
            
            // RHS: t -> s
            apply_Ts(t, q, v0, v1, n);
            apply_Ts(s, q, v1, v4, n);
            
            char tag[64]; sprintf(tag, "Comm s=%d,t=%d", s, t);
            if (!check_vectors(v3, v4, n, tag)) fails++;
            tests++;
        }
    }

    if (fails == 0) printf("[SUCCESS] All %d Hecke relations satisfied.\n", tests);
    else printf("[FAILURE] %d/%d tests failed.\n", fails, tests);

    free(v0); free(v1); free(v2); free(v3); free(v4);
}

void print_specific_tableau(int one_based_index) {
    int idx = one_based_index - 1; // Convert to 0-based internal index

    if (idx < 0 || idx >= NUM_TABLEAUX) {
        printf("Error: Index %d out of bounds (Max: %d)\n", one_based_index, NUM_TABLEAUX);
        return;
    }

    printf("\n========================================\n");
    printf(" TABLEAU %d \n", one_based_index);
    printf("========================================\n");

    // 1. Prepare a Grid (Max 64x64 to be safe)
    int grid[64][64]; 
    memset(grid, 0, sizeof(grid));

    // 2. Fill Grid using the flat arrays
    // ROW_INDS[idx * N + k] is the row of number (k+1)
    uint8_t *rows = &ROW_INDS[idx * N];
    uint8_t *cols = &COL_INDS[idx * N];

    for (int k = 0; k < N; k++) {
        int r = rows[k] - 1; // arrays store 1-based coordinates
        int c = cols[k] - 1;
        grid[r][c] = k + 1;
    }

    // 3. Print based on PARTITION shape
    for (int r = 0; r < PARTITION_LEN; r++) {
        for (int c = 0; c < PARTITION[r]; c++) {
            printf("%4d ", grid[r][c]);
        }
        printf("\n");
    }
    printf("\n");
}

// ==========================================
// Part 5: Operator (1+s) Export
// ==========================================
// ==========================================
// Part 5: Operator (1+s) Export (TRANSPOSED)
// ==========================================
void export_operators(const char* filename) {
    printf("Exporting (1+s) Operators (Transposed) to %s...\n", filename);
    FILE *f = fopen(filename, "wb");
    if (!f) {
        printf("[Error] Could not open file for writing.\n");
        return;
    }

    // 1. Write Header: [Number of Generators, Matrix Dimension]
    int num_gens = N - 1;
    int dim = NUM_TABLEAUX;
    fwrite(&num_gens, sizeof(int), 1, f);
    fwrite(&dim, sizeof(int), 1, f);

    // 2. Loop through each generator s
    for (int s = 0; s < num_gens; s++) {
        // Placeholder for Number of Non-Zero (NNZ) entries
        long count_pos = ftell(f);
        int nnz = 0;
        fwrite(&nnz, sizeof(int), 1, f); 

        // Iterate through all tableaux u
        for (int u = 0; u < dim; u++) {
            int s_in_u = (DESCENTS[u] >> s) & 1;

            // --- Diagonal Entries ---
            // If s NOT in Descents(u), diagonal is 2. (Symmetric, so no swap needed)
            if (!s_in_u) {
                int val = 2;
                fwrite(&u, sizeof(int), 1, f); // Row
                fwrite(&u, sizeof(int), 1, f); // Col
                fwrite(&val, sizeof(int), 1, f); // Val
                nnz++;
            }

            // --- Off-Diagonal Entries ---
            // If s in Des(u) and NOT in Des(v), we have an edge u -> v
            if (s_in_u) {
                VertexFlat *vert = &FLAT_GRAPH[u];
                for (int i = 0; i < vert->size; i++) {
                    int v = vert->edges[i].target;
                    
                    int s_in_v = (DESCENTS[v] >> s) & 1;

                    // Condition: s in u, s NOT in v
                    if (!s_in_v) {
                        int w = vert->edges[i].weight;

                        // FIX: Write (u, v) instead of (v, u)
                        // Row = u (Source), Col = v (Target)
                        fwrite(&u, sizeof(int), 1, f); 
                        fwrite(&v, sizeof(int), 1, f); 
                        fwrite(&w, sizeof(int), 1, f); 
                        nnz++;
                    }
                }
            }
        }

        // Go back and write the actual NNZ count
        long end_pos = ftell(f);
        fseek(f, count_pos, SEEK_SET);
        fwrite(&nnz, sizeof(int), 1, f);
        fseek(f, end_pos, SEEK_SET);
    }

    fclose(f);
    printf("Operator export complete.\n");
}

int main() {
    //clock_t start = clock();
    generate_tableaux();
    //print_specific_tableau(390948);
    //print_specific_tableau(13817814);

    //print_specific_tableau(122108);
    //print_specific_tableau(122109);
    //print_specific_tableau(122110);
    //print_specific_tableau(158573);
    //print_specific_tableau(158574);
    //print_specific_tableau(158575);
    //print_specific_tableau(169265);
    //print_specific_tableau(169266);
    //print_specific_tableau(169267);
    compute_weighted_graph();
    //test_hecke_relations();
    //printf("Total Execution Time: %.2f s\n", (double)(clock()-start)/CLOCKS_PER_SEC);
    //export_binary("sparse_graph.bin");
    //export_adjacency_binary("sparse_adj_graph.bin");
    //free(COL_INDS); free(ROW_INDS); free(DESCENTS); free(TABLE);
    //free(HASH_VALUES); free(HASH_KEYS);
    //free(WEIGHT_CACHE); free(CACHE_TAGS);
    
    //for(int i=0; i<NUM_TABLEAUX; i++) free(FLAT_GRAPH[i].edges);
    //free(FLAT_GRAPH);
    export_operators("operators.bin");
    return 0;
}