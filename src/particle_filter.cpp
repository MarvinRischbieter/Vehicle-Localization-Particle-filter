/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h>
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
    // TODO: Set the number of particles. Initialize all particles to first position (based on estimates of
    //   x, y, theta and their uncertainties from GPS) and all weights to 1.
    // Add random Gaussian noise to each particle.
    // NOTE: Consult particle_filter.h for more information about this method (and others in this file).
    static default_random_engine gen;
    num_particles = 100; // init number of particles to use
    // Normal distributions of x, y and theta.
    normal_distribution<double> dist_x(x, std[0]);
    normal_distribution<double> dist_y(y, std[1]);
    normal_distribution<double> dist_theta(theta, std[2]);
    // Pre-allocate vectors for the particles and their weight
    particles.resize(num_particles);
    weights.resize(num_particles);
    for (int i = 0; i < num_particles; i++) {
        particles[i].id = i;
        particles[i].x = dist_x(gen);
        particles[i].y = dist_y(gen);
        particles[i].theta = dist_theta(gen);
        particles[i].weight = 1.0/num_particles;
    }
    is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[],
                                double velocity, double yaw_rate) {
    // TODO: Add measurements to each particle and add random Gaussian noise.
    // NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
    //  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
    //  http://www.cplusplus.com/reference/random/default_random_engine/
    static default_random_engine gen;
    normal_distribution<double> dist_x(0.0, std_pos[0]);
    normal_distribution<double> dist_y(0.0, std_pos[1]);
    normal_distribution<double> dist_theta(0.0, std_pos[2]);
    for (Particle& particle: particles) {
        if (fabs(yaw_rate) > 0.001) {
            const double theta_new = particle.theta + yaw_rate * delta_t;
            particle.x += velocity / yaw_rate * (sin(theta_new) - sin(
                                  particle.theta));
            particle.y += velocity / yaw_rate * (-cos(theta_new) + cos(
                                  particle.theta));
            particle.theta = theta_new;
        }
        else {  // Special case
            particle.x += velocity * delta_t * cos(particle.theta);
            particle.y += velocity * delta_t * sin(particle.theta);
        }
        // Random Gaussian noise
        particle.x += dist_x(gen);
        particle.y += dist_y(gen);
        particle.theta += dist_theta(gen);
    }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
                                   const std::vector<LandmarkObs> &observations,
                                   const Map &map_landmarks) {
    const double sigma_x = std_landmark[0];
    const double sigma_y = std_landmark[1];
    const double sigma_x2 = sigma_x * sigma_x;
    const double sigma_y2 = sigma_y * sigma_y;
    for (Particle& particle: particles) {
        const float x = particle.x;
        const float y = particle.y;
        const double sin_theta = sin(particle.theta);
        const double cos_theta = cos(particle.theta);
        vector<double> weights_col = {};  // vector for weights accumulation
        for (const LandmarkObs& obs: observations) {
            // Observation coordinates transform
            LandmarkObs observation;
            observation.id = obs.id;
            observation.x = x + (obs.x * cos_theta) - (obs.y * sin_theta);
            observation.y = y + (obs.x * sin_theta) + (obs.y * cos_theta);
            // Nearest-neighbors data association
            Map::single_landmark_s nearest_lm;
            double nearest_distance = numeric_limits<double>::max();
            for (auto& lm: map_landmarks.landmark_list) {
                double distance = dist(lm.x_f, lm.y_f, observation.x, observation.y);
                if (distance < nearest_distance && distance < sensor_range) {
                    nearest_distance = distance;
                    nearest_lm = lm;
                }
            }
            if (nearest_distance < numeric_limits<double>::max()) {  // If we found
                const double dx = observation.x - nearest_lm.x_f;
                const double dy = observation.y - nearest_lm.y_f;
                weights_col.push_back(dx * dx / sigma_x2 + dy * dy / sigma_y2);
            }
            else {
                weights_col.clear();
                break;
            }
        }
        if (weights_col.empty()) {
            particle.weight = 0.0;
        }
        else {
            particle.weight = exp(-0.5*accumulate(weights_col.begin(),
                                                  weights_col.end(), 0.0));
        }
    }
    // Weights normalization to sum(weights)=1
    double weights_sum = 0;
    for(Particle& p: particles) weights_sum += p.weight;
    const double norm_weight = 2 * weights_sum * M_PI * sigma_x * sigma_y;
    for (int i = 0; i < num_particles; i++) {
        particles[i].weight /= norm_weight;
        weights[i] = particles[i].weight;
    }
}

void ParticleFilter::resample() {
    // TODO: Resample particles with replacement with probability proportional to their weight.
    // NOTE: You may find std::discrete_distribution helpful here.
    //   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
    static default_random_engine gen;
    discrete_distribution<> dist_idx(weights.begin(), weights.end());
    vector<Particle> resampled(num_particles);
    for (int i = 0; i < num_particles; i++) {
        resampled[i] = particles[dist_idx(gen)];
    }
    particles = resampled;

}

Particle ParticleFilter::SetAssociations(Particle& particle,
        const std::vector<int>& associations,
        const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    // particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates
    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
    return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
    vector<int> v = best.associations;
    stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}

string ParticleFilter::getSenseX(Particle best)
{
    vector<double> v = best.sense_x;
    stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
    vector<double> v = best.sense_y;
    stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
