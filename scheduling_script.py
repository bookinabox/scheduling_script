"""Creates a shift scheduling problem and solves it."""

from absl import app
from absl import flags

from google.protobuf import text_format
from ortools.sat.python import cp_model
import pandas as pd
import json

# idk why I put this up here. script will break if != 2
#(since rn assuming mostly consecutive shifts)
SHIFTS_PER_TUTOR = 2
# "Normalizes" tutors per shift.
EQUALITY_WEIGHT = 100
# it may be infeasible to solve if we
# force an average number of people per shift
# This relaxes the number of people per shift
SHIFT_RELAX = 2 


# Arbitrary weights. Might require some tuning
FIRST_CHOICE_WEIGHT = 3
FIRST_CHOICE_COL = 'First Preferred Tutoring Time Slot (Example Format: Monday 9-11)'
SECOND_CHOICE_WEIGHT = 2
SECOND_CHOICE_COL = 'Second Preferred Tutoring Time Slot'
THIRD_CHOICE_WEIGHT = 1
THIRD_CHOICE_COL = 'Third Preferred Tutoring  Time Slot '



filename = 'responses_officers.csv'

#### Values to make toggeable
display = True # prints out to console formatted

# Ensures all people will get at least one preference
# May make script break, but can be fixed by increasing SHIFT_RELAX (most of the time)
at_least_one_preference = True

# csv or json
output_format = 'csv'

# This should be cleaned up
# But spaghetti due to legacy things
days_map = {'Monday' : 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3, "Friday": 4}
days_map_short = {'Monday': 'M', 'Tuesday': 'T', 'Wednesday': 'W', 'Thursday': 'R', 'Friday': 'F'}
days_map_inverse = {0: 'Monday', 1 : "Tuesday", 2: "Wednesday", 3:"Thursday", 4: "Friday"}
shifts = ['9-10','10-11','11-12','12-1','1-2','2-3','3-4','4-5']
shifts_map = {'9-11' : 0, '10-12' : 1, '11-1' : 2, '12-2': 3, '1-3': 4, '2-4': 5, '3-5':6}
shifts_map_inverse = {0: '9-11', 1: '10-12', 2: '11-1', 3: '12-2', 4: '1-3', 5:'2-4', 6:'3-5'}

FLAGS = flags.FLAGS
flags.DEFINE_string('output_proto', '',
                    'Output file to write the cp_model proto to.')
flags.DEFINE_string('params', 'max_time_in_seconds:100.0',
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
    
    # Force consecutive Shifts (only works for 2. TODO: Generalize)
    # Redundant constraint if preferences are enforced
    if not at_least_one_preference:
        for e in range(num_employees):
            for d in range(num_days):
                for s in range(1,num_shifts-1):
                    model.AddBoolOr([work[(e, s-1, d)], work[(e, s, d)].Not(), work[(e, s+1, d)]])
                model.Add(work[e, 1, d]==1).OnlyEnforceIf(work[e, 0, d])
                model.Add(work[e, num_shifts-2, d]==1).OnlyEnforceIf(work[e, num_shifts-1, d])

    ##### Soft Constraints

    # Normalize number of people per shift (assumes equal workload per shift)
    min_demand = (num_employees * SHIFTS_PER_TUTOR) // (num_shifts * num_days) - SHIFT_RELAX
    delta = {}
    delta_s = {}
    for s in range(num_shifts):
        for d in range(num_days):
            works = [work[e, s, d] for e in range(num_employees)]
            delta[(s, d)] = model.NewIntVar(0, num_employees, f'delta_{s}_{d}')
            model.Add(delta[s, d] == sum(works) - min_demand)
            delta_s[(s, d)] = model.NewIntVar(0, num_employees**2, f'delta_{s}_{d}')
            # Square creates quadratic loss function
            model.AddMultiplicationEquality(delta_s[(s,d)], [delta[(s,d)], delta[(s,d)]])

    # Generate Tutor Preferences
    tutor_preferences = []
    tutor_preference_weights = []

    for index, row in responses_df.iterrows():
        pref_1_day, pref_1_shift = convert_preference_to_tuple(row[FIRST_CHOICE_COL])
        pref_1 = work[index, pref_1_shift, pref_1_day]
        tutor_preferences.append(pref_1)
        tutor_preference_weights.append(FIRST_CHOICE_WEIGHT)
        model.Add(work[index, pref_1_shift+1, pref_1_day]==1).OnlyEnforceIf(pref_1)

        pref_2_day, pref_2_shift = convert_preference_to_tuple(row[SECOND_CHOICE_COL])
        pref_2 = work[index, pref_2_shift, pref_2_day]
        tutor_preferences.append(pref_2)
        tutor_preference_weights.append(SECOND_CHOICE_WEIGHT)
        model.Add(work[index, pref_2_shift+1, pref_2_day]==1).OnlyEnforceIf(pref_2)

        pref_3_day, pref_3_shift = convert_preference_to_tuple(row[THIRD_CHOICE_COL])
        pref_3 = work[index, pref_3_shift, pref_3_day]
        tutor_preferences.append(pref_3)
        tutor_preference_weights.append(THIRD_CHOICE_WEIGHT)
        model.Add(work[index, pref_3_shift+1, pref_3_day]==1).OnlyEnforceIf(pref_3)
        
        # Get at least one preference.
        if at_least_one_preference:
            model.AddBoolOr([pref_1, pref_2, pref_3])

    model.Maximize(sum(tutor_preferences[i] * tutor_preference_weights[i] for i in range(len(tutor_preferences)))
                    - sum(delta_s[s, d] for s in range(num_shifts) for d in range(num_days)) * EQUALITY_WEIGHT
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
        if display:
            print()
            for d in range(num_days):
                print(list(days_map.keys())[d] + '------------------------------------------')
                for s in range(num_shifts):
                    print(shifts[s])
                    for e in range(num_employees):
                        if solver.BooleanValue(work[e, s, d]):
                            print(f"\t{responses_df['Email Address'].iloc[e]}")

        final_schedule = {}
        if output_format == 'json':
            for day in days_map.keys():
                for s, shift in enumerate(shifts):
                    final_schedule[f"{day} {shift}"] = []
                    for e in range(num_employees):
                        if solver.BooleanValue(work[e, s, days_map[day]]):
                            final_schedule[f"{day} {shift}"].append(f"{responses_df['First Name'].iloc[e]} {responses_df['Last Name'].iloc[e]}")
            json_object = json.dumps(final_schedule, indent=4)
            with open("tentative_final_hours.json", "w") as f:
                f.write(json_object)
        elif output_format == 'csv':
            # Format currently ingested by website
            # I like json more :(
            tutoring_script_data = []
            for day in days_map.keys():
                for e in range(num_employees):
                    for s in range(len(shifts)-1):
                        if solver.BooleanValue(work[e, s, days_map[day]]) and solver.BooleanValue(work[e, s+1, days_map[day]]):
                            tutoring_script_data.append({"Day of Week":days_map_short[day], "Time-Time":shifts_map_inverse[s], "First Name":responses_df['First Name'].iloc[e], "Last Name":responses_df['Last Name'].iloc[e], "Zoom Link":None})
            
            df = pd.DataFrame(tutoring_script_data)
            df.to_csv('tutoring_script_officer_format_data.csv')
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
    responses_df = pd.read_csv(filename)
    solve_shift_scheduling(FLAGS.params, FLAGS.output_proto, responses_df)


if __name__ == '__main__':
    app.run(main)