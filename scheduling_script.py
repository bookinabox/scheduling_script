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

FLAGS = flags.FLAGS
flags.DEFINE_string('output_proto', '',
                    'Output file to write the cp_model proto to.')
flags.DEFINE_string('params', 'max_time_in_seconds:10.0',
                    'Sat solver parameters.')

def negated_bounded_span(works, start, length):
    """Filters an isolated sub-sequence of variables assined to True.
  Extract the span of Boolean variables [start, start + length), negate them,
  and if there is variables to the left/right of this span, surround the span by
  them in non negated form.
  Args:
    works: a list of variables to extract the span from.
    start: the start to the span.
    length: the length of the span.
  Returns:
    a list of variables which conjunction will be false if the sub-list is
    assigned to True, and correctly bounded by variables assigned to False,
    or by the start or end of works.
  """
    sequence = []
    # Left border (start of works, or works[start - 1])
    if start > 0:
        sequence.append(works[start - 1])
    for i in range(length):
        sequence.append(works[start + i].Not())
    # Right border (end of works or works[start + length])
    if start + length < len(works):
        sequence.append(works[start + length])
    return sequence

def add_soft_sequence_constraint(model, works, hard_min, soft_min, min_cost,
                                 soft_max, hard_max, max_cost, prefix):
    """Sequence constraint on true variables with soft and hard bounds.
  This constraint look at every maximal contiguous sequence of variables
  assigned to true. If forbids sequence of length < hard_min or > hard_max.
  Then it creates penalty terms if the length is < soft_min or > soft_max.
  Args:
    model: the sequence constraint is built on this model.
    works: a list of Boolean variables.
    hard_min: any sequence of true variables must have a length of at least
      hard_min.
    soft_min: any sequence should have a length of at least soft_min, or a
      linear penalty on the delta will be added to the objective.
    min_cost: the coefficient of the linear penalty if the length is less than
      soft_min.
    soft_max: any sequence should have a length of at most soft_max, or a linear
      penalty on the delta will be added to the objective.
    hard_max: any sequence of true variables must have a length of at most
      hard_max.
    max_cost: the coefficient of the linear penalty if the length is more than
      soft_max.
    prefix: a base name for penalty literals.
  Returns:
    a tuple (variables_list, coefficient_list) containing the different
    penalties created by the sequence constraint.
  """
    cost_literals = []
    cost_coefficients = []

    # Forbid sequences that are too short.
    for length in range(1, hard_min):
        for start in range(len(works) - length + 1):
            model.AddBoolOr(negated_bounded_span(works, start, length))

    # Penalize sequences that are below the soft limit.
    if min_cost > 0:
        for length in range(hard_min, soft_min):
            for start in range(len(works) - length + 1):
                span = negated_bounded_span(works, start, length)
                name = ': under_span(start=%i, length=%i)' % (start, length)
                lit = model.NewBoolVar(prefix + name)
                span.append(lit)
                model.AddBoolOr(span)
                cost_literals.append(lit)
                # We filter exactly the sequence with a short length.
                # The penalty is proportional to the delta with soft_min.
                cost_coefficients.append(min_cost * (soft_min - length))

    # Penalize sequences that are above the soft limit.
    if max_cost > 0:
        for length in range(soft_max + 1, hard_max + 1):
            for start in range(len(works) - length + 1):
                span = negated_bounded_span(works, start, length)
                name = ': over_span(start=%i, length=%i)' % (start, length)
                lit = model.NewBoolVar(prefix + name)
                span.append(lit)
                model.AddBoolOr(span)
                cost_literals.append(lit)
                # Cost paid is max_cost * excess length.
                cost_coefficients.append(max_cost * (length - soft_max))

    # Just forbid any sequence of true variables with length hard_max + 1
    for start in range(len(works) - hard_max):
        model.AddBoolOr(
            [works[i].Not() for i in range(start, start + hard_max + 1)])
    return cost_literals, cost_coefficients


def solve_shift_scheduling(params, output_proto):
    """Solves the shift scheduling problem."""
    # Data
    num_employees = 100
    shifts = ['9-10', '10-11', '11-12', '12-1', '1-2', '2-3', '3-4', '4-5']

    num_days = 5
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
                
        model.Add(sum(total_shifts) == 2)
    
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

    # Normalize number of people per shift


    # Normalize class variety (probably not that important)


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
        days = ['Mon', 'Tues', 'Wed', 'Thurs', 'Fri']
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