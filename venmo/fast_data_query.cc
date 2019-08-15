 #include <iostream>
#include <fstream>
#include <map>
#include <set>
#include <algorithm>
#include <cstdlib>
#include <vector>
#include <random>
#include <unordered_set>
#include <cstdio>

#define MAX_UID   40000000
#define YEAR      31536000

using namespace std;

typedef struct {
  int timestamp;
  int charge;
  int neighbor;
} adj_t;

typedef struct {
  int pay_id;
  int actor;
  int timestamp;
  int charge;
  int target;
} edge_t;

typedef struct {
  int uid;
  int timestamp;
} regis_t;

typedef struct {
  int pay_id;
  int timestamp;
  int node;
  vector<int> candidates;
  vector<int> selected_size;
  vector<int> group_size;
  vector<vector<int>> features;
  int label;
} train_data_t;

vector<vector<adj_t>> blank_adj_lists(int num_nodes) {
  vector<vector<adj_t>> result;
  for (int i = 0; i < num_nodes; ++i) {
    vector<adj_t> v;
    result.push_back(v);
  }
  return result;
}

void add_edge(int pay_id, int timestamp, int actor, int charge, int target,
              vector<vector<adj_t>> &adj_lists, vector<edge_t> &edge_list) {
  adj_t a0, a1;
  a0.timestamp = timestamp;
  a0.neighbor = target;
  a0.charge = charge;
  a1.timestamp = timestamp;
  a1.neighbor = actor;
  a1.charge = 1-charge;
  adj_lists[actor].push_back(a0);
  adj_lists[target].push_back(a1);
  edge_t e;
  e.pay_id = pay_id;
  e.actor = actor;
  e.charge = charge;
  e.target = target;
  e.timestamp = timestamp;
  edge_list.push_back(e);
}

void process_file(const char *filename, vector<vector<adj_t>> &adj_lists, vector<edge_t> &edge_list) {
  // cout << '\r' << filename;
  ifstream edges(filename);
  int pay_id, timestamp, actor, charge, target;
  while (edges >> pay_id >> timestamp >> actor >> charge >> target) {
    add_edge(pay_id, timestamp, actor, charge, target, adj_lists, edge_list);
  }
}

void process_regis_file(const char *filename, vector<regis_t> &regis_dates) {
  ifstream regis(filename);
  int uid, timestamp;
  while (regis >> uid >> timestamp) regis_dates.push_back({uid, timestamp});
}

template <class T>
size_t random_fill(vector<T> &target, vector<T> &source, size_t max_size) {
  size_t source_size = source.size();
  if (max_size >= source_size) {
    target.insert(target.end(), source.begin(), source.end());
    return source_size;
  }
  random_shuffle(source.begin(), source.end());
  target.insert(target.end(), source.begin(), source.begin() + int(max_size));
  return max_size;
}

template <class T>
void fill(vector<T> &v, T element, size_t size) {
  for (size_t i = 0; i < size; ++i) v.push_back(element);
}

template <class T>
bool cmp(T t, int timestamp) {
  return (t.timestamp < timestamp);
}


int randint(const int range_from, const int range_to) {
  random_device                  rand_dev;
  mt19937                        generator(rand_dev());
  uniform_int_distribution<int>  distr(range_from, range_to);
  return distr(generator);
}

bool flip(float p) {
  random_device                     rand_dev;
  mt19937                           generator(rand_dev());
  uniform_real_distribution<float>  distr(1.0, 2.0);
  return distr(generator) < 1+p;
}

void uniform_sampling(int timestamp, int actor, int charge, int target, int time_window,
                      int num_neg_samples, vector<regis_t> &registered_users,
                      vector<int> &local_candidates, vector<int> &selected_size, vector<int> &group_size) {

  int node  = charge ? target : actor;
  int label = charge ? actor : target;
  set<int> other_candidates;
  size_t pos = lower_bound(registered_users.begin(), registered_users.end(), timestamp, cmp<regis_t>) - registered_users.begin();

  while (other_candidates.size() < size_t(num_neg_samples)) {
    int sampled_user = registered_users[randint(0, pos-1)].uid;
    if (sampled_user != node && sampled_user != label) {
      other_candidates.insert(sampled_user);
    }
  }
  local_candidates.push_back(label);
  local_candidates.insert(local_candidates.end(), other_candidates.begin(), other_candidates.end());
  fill(selected_size, num_neg_samples+1, size_t(num_neg_samples)+1);
  fill(group_size,  int(pos - 1), size_t(num_neg_samples)+1);
}

void stratified_sampling(int timestamp, int actor, int charge, int target, int time_window,
                         int num_neg_samples, int num_n1_samples, int num_n2_samples,
                         vector<vector<adj_t>> &adj_lists, vector<regis_t> &registered_users,
                         vector<int> &local_candidates, vector<int> &selected_size,
                         vector<int> &group_size) {

  int node  = charge ? target : actor;
  int label = charge ? actor : target;
  
  int label_group = 0;

  set<int> neighbor_1, neighbor_2;
  
  // Count only degrees after the look-up window

  // Add 1st degree neighborhood
  for (adj_t a : adj_lists[node]) {
    if (a.timestamp < timestamp - time_window) continue;
    if (a.timestamp >= timestamp) break;
    if (a.neighbor != label) {
      neighbor_1.insert(a.neighbor);
    } else if (a.neighbor == label) {
      label_group = 1;
    }
  }

  // Add 2nd degree neighborhood
  for (adj_t a : adj_lists[node]) {
    if (a.timestamp < timestamp - time_window) continue;
    if (a.timestamp >= timestamp) break;
    if (a.charge == 0) {
      for (adj_t a2 : adj_lists[a.neighbor]) {
        if (a2.timestamp < timestamp - time_window) continue;
        if (a2.timestamp >= timestamp) break;
        if (neighbor_1.count(a2.neighbor) == 0 && a2.charge == 0 &&
            a2.neighbor != node && a2.neighbor != label) {
          neighbor_2.insert(a2.neighbor);
        } else if (a2.neighbor == label && label_group == 0) {
          label_group = 2;
        }
      }
    }  
  }

  // Convert to vector
  vector<int> group1(neighbor_1.begin(), neighbor_1.end());
  vector<int> group2(neighbor_2.begin(), neighbor_2.end());
  set<int>    other_candidates;

  // The label is the first candidate 
  local_candidates.push_back(label);

  // Fill the target vector with randomly selected candidates in strata n1, n2
  size_t n1  = random_fill(local_candidates, group1, num_n1_samples);
  size_t n2  = random_fill(local_candidates, group2, num_n2_samples);
  size_t n3  = num_neg_samples - n1 - n2;
  size_t pos = lower_bound(registered_users.begin(), registered_users.end(), timestamp, cmp<regis_t>) - registered_users.begin();

  // Fill other spots left in the target vector
  while (other_candidates.size() < n3) {
    int sampled_user = registered_users[randint(0, pos-1)].uid;
    if (neighbor_1.count(sampled_user) == 0 && neighbor_2.count(sampled_user) == 0 &&
        sampled_user != node && sampled_user != label) {
      other_candidates.insert(sampled_user);
    }
  }
  local_candidates.insert(local_candidates.end(), other_candidates.begin(), other_candidates.end());

  switch (label_group) {
    case 1: {
      fill(selected_size, int(n1)+1, 1); // replace left 1 with n1+1, n2+1, or n3+1, depending on which group
      fill(selected_size, int(n1)+1, n1);
      fill(selected_size, int(n2), n2);
      fill(selected_size, int(n3), n3);
      fill(group_size, int(group1.size())+1, 1);
      fill(group_size, int(group1.size())+1, n1);
      fill(group_size, int(group2.size()), n2);
      fill(group_size, int(pos - 2 - group1.size() - group2.size()), n3);
      break;
    }
    case 2: {
      fill(selected_size, int(n2)+1, 1); // replace left 1 with n1+1, n2+1, or n3+1, depending on which group
      fill(selected_size, int(n1), n1);
      fill(selected_size, int(n2)+1, n2);
      fill(selected_size, int(n3), n3);
      fill(group_size, int(group2.size())+1, 1);
      fill(group_size, int(group1.size()), n1);
      fill(group_size, int(group2.size())+1, n2);
      fill(group_size, int(pos - 2 - group1.size() - group2.size()), n3);
      break;
    }
    default: {
      fill(selected_size, int(n3)+1, 1); // replace left 1 with n1+1, n2+1, or n3+1, depending on which group
      fill(selected_size, int(n1), n1);
      fill(selected_size, int(n2), n2);
      fill(selected_size, int(n3)+1, n3);
      fill(group_size, int(pos - 2 - group1.size() - group2.size())+1, 1);
      fill(group_size, int(group1.size()), n1);
      fill(group_size, int(group2.size()), n2);
      fill(group_size, int(pos - 2 - group1.size() - group2.size())+1, n3);
      break;
    }
  }
  // Record pool size for sampling weight correction
  
}

void extract_features(vector<int> &candidates, int timestamp, int actor, int charge, int target,
                      vector<vector<adj_t>> &adj_lists, int time_window, vector<vector<int>> &features) {

  int node = charge ? target : actor;
  unordered_set<int> my_friends;
  
  // Reference to adjacency list of the payer node
  vector<adj_t> &node_data = adj_lists[node];
  
  // Get the size of the adjacency list
  size_t node_data_size = node_data.size();
  
  // Find the offset location where the look-up begins
  size_t pos = lower_bound(node_data.begin(), node_data.end(),
                           timestamp - time_window, cmp<adj_t>) - node_data.begin();
  
  // Store the set of friends (payee), make sure it doesn't overflow, look up until current timestamp
  for (size_t i = pos; i < node_data_size && node_data[i].timestamp < timestamp; ++i) {
    if (node_data[i].charge == 0) my_friends.insert(node_data[i].neighbor);
  }
  
  // Iterate through candidates
  for (int candidate : candidates) {
    int x_in=0, x_out=0, i_paid_x=0, x_paid_me=0;
    unordered_set<int> x_friends, my_friends_that_paid_x;
    
    vector<adj_t> &candidate_data = adj_lists[candidate];
    size_t pos = lower_bound(candidate_data.begin(), candidate_data.end(),
                             timestamp - time_window, cmp<adj_t>) - candidate_data.begin();
    size_t candidate_data_size = candidate_data.size();
    
    // Iterate through candidate history
    for (size_t i = pos; i < candidate_data_size && candidate_data[i].timestamp < timestamp; ++i) {
      auto a = candidate_data[i];
      x_in += a.charge;
      x_out += (1-a.charge);
      if (a.neighbor == node) {
        i_paid_x += a.charge;
        x_paid_me += (1-a.charge);
      }
      x_friends.insert(a.neighbor);
      if (my_friends.count(a.neighbor) > 0 && a.charge > 0) my_friends_that_paid_x.insert(a.neighbor);
    }
    
    features.push_back(vector<int> ({x_in, x_out, int(x_friends.size()), i_paid_x, x_paid_me, int(my_friends_that_paid_x.size())}));
  }
  
}

void sample_from_adj_lists(int min_timestamp, int max_timestamp, int time_window, float sampling_rate,
                       int num_neg_samples, int num_n1_samples, int num_n2_samples,
                       vector<vector<adj_t>> &adj_lists, vector<edge_t> &edge_list,
                       vector<regis_t> &registered_users, vector<train_data_t> &data) {
  size_t pos = lower_bound(edge_list.begin(), edge_list.end(), min_timestamp, cmp<edge_t>) - edge_list.begin();
  size_t edge_list_size = edge_list.size();
  for (size_t i = pos; i < edge_list_size && edge_list[i].timestamp < max_timestamp; ++i) {
    if (!flip(sampling_rate)) continue;
    edge_t e = edge_list[i];
    vector<int> local_candidates, selected_size, group_size;
    vector<vector<int>> features;

    // Uniform or stratified sampling
    if ((num_n1_samples < 0) || (num_n2_samples < 0)) {
      uniform_sampling(e.timestamp, e.actor, e.charge, e.target, time_window,
                       num_neg_samples, registered_users, local_candidates, selected_size, group_size);
    }
    else {
      stratified_sampling(e.timestamp, e.actor, e.charge, e.target, time_window,
                          num_neg_samples, num_n1_samples, num_n2_samples,
                          adj_lists, registered_users, local_candidates, selected_size, group_size);
    }

    extract_features(local_candidates, e.timestamp, e.actor, e.charge, e.target,
                     adj_lists, time_window, features);
    data.push_back({e.pay_id, e.timestamp, e.charge ? e.target : e.actor, local_candidates,
                    selected_size, group_size, features, e.charge ? e.actor : e.target});
  }
}

void write_data(vector<train_data_t> &data, const char* filename) {
  ofstream out(filename);
  for (auto a : data) {
    size_t max_size = a.candidates.size();
    for (size_t i = 0; i < max_size; ++i) {
      out << a.pay_id << "," << a.timestamp << "," << a.node << "," << a.candidates[i];
      out << "," << (i == 0 ? 1 : 0) << "," << a.selected_size[i] << "," << a.group_size[i];
      for (auto f : a.features[i]) out << "," << f;
      out << endl;
    }
  }
}      

int main(int argc, char *argv[]) {

  vector<vector<adj_t>> adj_lists = blank_adj_lists(MAX_UID);
  vector<regis_t> registered_users;
  vector<edge_t> edge_list;
  process_regis_file("users.regisdate", registered_users);

  for (int i = 1; i < argc; ++i) process_file(argv[i], adj_lists, edge_list);
  // cout << endl;

  int min_timestamp, max_timestamp, time_window, num_neg_samples, num_n1_samples, num_n2_samples;
  float sampling_rate;
  
  while (cin >> min_timestamp >> max_timestamp >> time_window >>
                sampling_rate >> num_neg_samples >> num_n1_samples >> num_n2_samples) {

    char filename[64];
    sprintf(filename, "%d-%d-%d-%.4f-%d-%d-%d", min_timestamp,   max_timestamp,  time_window, sampling_rate,
                                                num_neg_samples, num_n1_samples, num_n2_samples);
    vector<train_data_t> data;

    sample_from_adj_lists(min_timestamp,   max_timestamp,  time_window, sampling_rate,
                          num_neg_samples, num_n1_samples, num_n2_samples,
                          adj_lists, edge_list, registered_users, data);

    write_data(data, filename);  
  }

  return 0;
}
