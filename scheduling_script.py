#!/usr/bin/env python3
# Copyright 2010-2021 Google LLC
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Modified from https://github.com/google/or-tools/blob/master/examples/python/shift_scheduling_sat.py

"""Creates a shift scheduling problem and solves it."""

from absl import app
from absl import flags

from google.protobuf import text_format
from ortools.sat.python import cp_model
import pandas as pd

days = ['Mon', 'Tues', 'Wed', 'Thurs', 'Fri']
shifts = ['9-10', '10-11', '11-12', '12-1', '1-2', '2-3', '3-4', '4-5']
SHIFTS_PER_TUTOR = 2 # TODO: Changing this will break everything for my consecutive shifts code. So... don't change it please.
EQUALITY_WEIGHT = 10 # "Normalizes" tutors per shift

# Arbitrary weights. Might require some tuning
FIRST_CHOICE_WEIGHT = 20
SECOND_CHOICE_WEIGHT = 10
THIRD_CHOICE_WEIGHT = 5


FLAGS = flags.FLAGS
flags.DEFINE_string('output_proto', '',
                    'Output file to write the cp_model proto to.')
flags.DEFINE_string('params', 'max_time_in_seconds:10.0',
                    'Sat solver parameters.')

def solve_shift_scheduling(params, output_proto):
    """Solves the shift scheduling problem."""
    # Data
    num_employees = 100
    num_days = len(days)
    num_shifts = len(shifts)

    model = cp_model.CpModel()

    # Create every possible employee-shift-day combination
    work = {}
    for e in range(num_employees):
        for s in range(num_shifts):
            for d in range(num_days):
                work[e, s, d] = model.NewBoolVar('work%i_%i_%i' % (e, s, d))

    #### Hard Constraints

    # Two shifts per tutor
    for e in range(num_employees):
        total_shifts = []
        for s in range(num_shifts):
            for d in range(num_days):
                total_shifts.append(work[e, s, d])
                
        model.Add(sum(total_shifts) == SHIFTS_PER_TUTOR)
    
    # At least 1 tutor per shift (Maybe this should be a soft constraint, we'll see)
    for d in range(num_days):
        for s in range(num_shifts):
            model.AddAtLeastOne(work[e, s, d] for e in range(num_employees)) 

    # Force consecutive Shifts
    for e in range(num_employees):
        for d in range(num_days):
            for s in range(1,num_shifts-1):
                #model.Add(work[e, s+1, d]==1).OnlyEnforceIf(work[e, s-1, d].Not(), work[e, s, d])
                model.AddBoolOr([work[(e, s-1, d)], work[(e, s, d)].Not(), work[(e, s+1, d)]])
            
            # Literal edge cases lol
            model.Add(work[e, 1, d]==1).OnlyEnforceIf(work[e, 0, d])
            model.Add(work[e, 0, d]==1).OnlyEnforceIf(work[e, 1, d])
            model.Add(work[e, num_shifts-1, d]==1).OnlyEnforceIf(work[e, num_shifts-2, d])
            model.Add(work[e, num_shifts-2, d]==1).OnlyEnforceIf(work[e, num_shifts-1, d])
    
    ##### Soft Constraints

    # Tutor Preferences
    tutor_preferences = []
    tutor_preference_weights = []

    #test = True
    #if test:
    #    tutor_preferences.append(work[1, 6, 4])
    #    tutor_preference_weights.append(10)


    # Normalize number of people per shift (assumes equal workload per shift)
    # TODO: modify to be based on shift traffic
    min_demand = num_employees * SHIFTS_PER_TUTOR // (num_shifts * num_days)
    worked = {}
    delta = model.NewIntVar(0, num_employees, "delta")
    for s in range(num_shifts):
        for d in range(5):
            works = [work[e, s, d] for e in range(num_employees)]
            worked[(s, d)] = model.NewIntVar(min_demand, num_employees, '')
            model.Add(worked[s, d] == sum(works))
            model.Add(worked[s, d] >= min_demand - delta)
            model.Add(worked[s, d] <= min_demand - delta)
    
    # TODO: Normalize class variety (probably not that important)
    
    model.Maximize(sum(tutor_preferences[i] * tutor_preference_weights[i] for i in range(len(tutor_preferences)))
                    - delta * EQUALITY_WEIGHT
                    )

    if output_proto:
        print('Writing proto to %s' % output_proto)
        with open(output_proto, 'w') as text_file:
            text_file.write(str(model))

    # Solve the model.
    solver = cp_model.CpSolver()
    if params:
        text_format.Parse(params, solver.parameters)
    solution_printer = cp_model.ObjectiveSolutionPrinter()
    status = solver.Solve(model, solution_printer)

    # Print solution.
    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        
        print()
        for d in range(num_days):
            print(days[d] + '------------------------------------------')
            for s in range(num_shifts):
                print(shifts[s])
                for e in range(num_employees):
                    if solver.BooleanValue(work[e, s, d]):
                        print('\t worker %i' % (e))

    print()
    print('Statistics')
    print('  - status          : %s' % solver.StatusName(status))
    print('  - conflicts       : %i' % solver.NumConflicts())
    print('  - branches        : %i' % solver.NumBranches())
    print('  - wall time       : %f s' % solver.WallTime())


def main(_=None):
    solve_shift_scheduling(FLAGS.params, FLAGS.output_proto)


if __name__ == '__main__':
    app.run(main)