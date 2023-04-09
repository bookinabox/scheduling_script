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

#days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
days_map = {'Monday' : 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3, "Friday": 4}
shifts = ['9-10','10-11','11-12','12-1','1-2','2-3','3-4','4-5']
shifts_map = {'9-11' : 0, '10-12' : 1, '11-1' : 2, '12-2': 3, '1-3': 4, '2-4': 5, '3-5':6}
SHIFTS_PER_TUTOR = 2 # TODO: Changing this will break everything for my consecutive shifts code. So... don't change it please.
EQUALITY_WEIGHT = 10 # "Normalizes" tutors per shift

# Arbitrary weights. Might require some tuning
FIRST_CHOICE_WEIGHT = 20
FIRST_CHOICE_COL = 'First Preferred Tutoring Time Slot (2 hour blocks only; Example Format: Monday 9-11)'
SECOND_CHOICE_WEIGHT = 10
SECOND_CHOICE_COL = 'Second Preferred Tutoring Time Slot'
THIRD_CHOICE_WEIGHT = 5
THIRD_CHOICE_COL = 'Third Preferred Tutoring Time Slot'


FLAGS = flags.FLAGS
flags.DEFINE_string('output_proto', '',
                    'Output file to write the cp_model proto to.')
flags.DEFINE_string('params', 'max_time_in_seconds:10.0',
                    'Sat solver parameters.')

def solve_shift_scheduling(params, output_proto, responses_df):
    """Solves the shift scheduling problem."""
    # Data
    num_employees = len(responses_df)
    num_days = len(days_map)
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

    # Force consecutive Shifts (only works for 2. TODO: Generalize)
    for e in range(num_employees):
        for d in range(num_days):
            for s in range(1,num_shifts-1):
                model.AddBoolOr([work[(e, s-1, d)], work[(e, s, d)].Not(), work[(e, s+1, d)]])
            
            # Literal edge cases lol
            model.Add(work[e, 1, d]==1).OnlyEnforceIf(work[e, 0, d])
            model.Add(work[e, 0, d]==1).OnlyEnforceIf(work[e, 1, d])
            model.Add(work[e, num_shifts-1, d]==1).OnlyEnforceIf(work[e, num_shifts-2, d])
            model.Add(work[e, num_shifts-2, d]==1).OnlyEnforceIf(work[e, num_shifts-1, d])
    
    ##### Soft Constraints

    # Normalize number of people per shift (assumes equal workload per shift)
    # TODO: modify to be based on shift traffic, rather than uniform
    min_demand = num_employees * SHIFTS_PER_TUTOR // (num_shifts * num_days)
    print(min_demand)
    worked = {}
    delta = model.NewIntVar(0, num_employees, "delta")
    for s in range(num_shifts):
        for d in range(num_days):
            works = [work[e, s, d] for e in range(num_employees)]
            worked[(s, d)] = model.NewIntVar(min_demand, num_employees, '')
            model.Add(worked[s, d] == sum(works))
            model.Add(worked[s, d] >= min_demand - delta)
    
    # Generate Tutor Preferences
    tutor_preferences = []
    tutor_preference_weights = []

    for index, row in responses_df.iterrows():
        pref_1_day, pref_1_shift = convert_preference_to_tuple(row[FIRST_CHOICE_COL])
        tutor_preference_weights.append(work[index, pref_1_shift, pref_1_day])
        tutor_preference_weights.append(FIRST_CHOICE_WEIGHT)

        pref_2_day, pref_2_shift = convert_preference_to_tuple(row[SECOND_CHOICE_COL])
        tutor_preference_weights.append(work[index, pref_2_shift, pref_2_day])
        tutor_preference_weights.append(SECOND_CHOICE_WEIGHT)

        pref_3_day, pref_3_shift = convert_preference_to_tuple(row[THIRD_CHOICE_COL])
        tutor_preference_weights.append(work[index, pref_3_shift, pref_3_day])
        tutor_preference_weights.append(THIRD_CHOICE_WEIGHT)


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
            print(list(days_map.keys())[d] + '------------------------------------------')
            for s in range(num_shifts):
                print(shifts[s])
                for e in range(num_employees):
                    if solver.BooleanValue(work[e, s, d]):
                        print(f"\t{responses_df['Email Address'].iloc[e]}")

        final_schedule = pd.DataFrame()

        for day in days_map.keys():
            for shift in shifts:
                final_schedule[f"{day} {shift}"] = []

    print()
    print('Statistics')
    print('  - status          : %s' % solver.StatusName(status))
    print('  - conflicts       : %i' % solver.NumConflicts())
    print('  - branches        : %i' % solver.NumBranches())
    print('  - wall time       : %f s' % solver.WallTime())

def convert_preference_to_tuple(preference : str):
    day = days_map[preference.split(" ")[0]]
    shift = shifts_map[preference.split(" ")[1]]
    return day, shift

    


def main(_=None):

    # Set up dataframe structure
    responses_df = pd.read_csv("responses.csv")
    solve_shift_scheduling(FLAGS.params, FLAGS.output_proto, responses_df)


if __name__ == '__main__':
    app.run(main)