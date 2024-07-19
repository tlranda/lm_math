import json
import re
import time
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

def load_json_data_as_kvs(filename):
    """
        Open this json file and return pre-separated lists of its keys and values
        Not dictionary views, we're going to rearrange this data a lot
    """
    with open(filename,'r') as f:
        data = json.load(f)
        keys = list(data.keys())
        values = list(data.values())
        values = pd.DataFrame(values)
        # Start by counting the number of items in each category
        try:
            raw_values = values.map(lambda xx: xx['observe'])
            out_of = values.map(lambda xx: xx['out_of'])
        except AttributeError:
            raw_values = values.applymap(lambda xx: xx['observe'])
            out_of = values.applymap(lambda xx: xx['out_of'])

        ### Update ratios to be relative where appropriate
        # Regex-following (X/100%)
        ### Reduce Parseable & Correct by Memorized-Verbatim, as my experiment isn't well designed to differentiate on that
        # Parseability (X-Verbatim/100%-Verbatim)
        raw_values['parseable'] -= raw_values['memorized_verbatim']
        out_of['parseable'] -= raw_values['memorized_verbatim']
        # Correct (X-Verbatim/Parseable)
        raw_values['correct'] -= raw_values['memorized_verbatim']
        out_of['correct'] = raw_values['parseable']
        ### Mutually exclusive with Memorized-Verbatim
        # Memorized-Copycat (X/Parseable)
        out_of['memorized_copycat'] = raw_values['parseable']
        # Drop the verbatim column
        raw_values = raw_values.drop(columns=['memorized_verbatim'])
        out_of = out_of.drop(columns=['memorized_verbatim'])
        if 'special' not in filename:
            # Clean the name up
            raw_values = raw_values.rename(columns={"memorized_copycat": "memorized"})
            out_of = out_of.rename(columns={'memorized_copycat': 'memorized'})
            #raw_values = raw_values.drop(columns=['memorized_copycat'])
            #out_of = out_of.drop(columns=['memorized_copycat'])
        else:
            # Clean the name up
            raw_values = raw_values.rename(columns={"memorized_copycat": "memorized"})
            out_of = out_of.rename(columns={'memorized_copycat': 'memorized'})
        values = raw_values / out_of
    return keys, values

def pretty_fn_name(name):
    """
        Make function names a bit easier to read from the keys
    """
    if name.startswith('<built-in function'):
        return name[len('<built-in function '):-1]
    elif name.startswith('<function'):
        return name[len('<function '):name.rindex('at')]
    return name

def parse_keys(keys):
    """
        Use regex to determine the field values for keys even though the function references are non-eval-able
    """
    # From start of the string, skip the '(' literal and capture everything up until the first comma
    op_regex = re.compile(r"^\((.*?),")
    op_list = []
    # Skip over to the max_digits field and grab that number
    digits_regex = re.compile(r"^.*'max_digits:([0-9]+)'")
    digits_list = []
    # Skip to these boolean fields and grab their boolean values
    negative_regex = re.compile(r"^.*'include_negative:(False|True)'")
    negative_list = []
    feature_regex = re.compile(r"^.*'feature_expression:(False|True)'")
    feature_list = []
    # Complicated parse: Determine which execution (with_context=True, use_both=True, or NEITHER)
    # BOTH: 2 | CONTEXT: 1 | SYSTEM: 0
    use_both_regex = re.compile(r"^.*'use_both:(True)'")
    use_system_regex = re.compile(r"^.*'with_context:(False)'")
    execution_type = []
    for k in keys:
        op_list.append(pretty_fn_name(op_regex.match(k).groups()[0]))
        digits_list.append(int(digits_regex.match(k).groups()[0]))
        negative_list.append(negative_regex.match(k).groups()[0] == 'True')
        feature_list.append(feature_regex.match(k).groups()[0] == 'True')
        # Determine kind of execution by checking for each possibility
        if use_both_regex.match(k) is not None:
            execution_type.append(2)
        elif use_system_regex.match(k) is not None:
            execution_type.append(0)
        else:
            execution_type.append(1)
    key_info = {
                # Plot-separators
                'operations': np.array(op_list),
                # Or y-axis splitter
                'executions': np.array(execution_type),
                # X-axis
                'digits': np.array(digits_list),
                'negatives': np.array(negative_list),
                # Line separators
                'features': np.array(feature_list),
                }
    return key_info

def aggregate(key_info, operators, executions):
    """
        Aggregate plot data by indices
        Split plots between different operators and executions
    """
    plots = []
    for operator in operators:
        this_op = np.where(key_info['operations'] == operator)[0]
        """
        for execution in executions:
            this_execution = np.where(key_info['executions'] == execution)[0]
            combined = np.intersect1d(this_op,this_execution)
            if len(combined) > 0:
                plots.append(combined)
            else:
                print(f"There are no combinations of {operator} and {execution}")
        """
        if len(this_op) > 0:
            plots.append(this_op)
    return plots

def plot_1(key_info,values,first_plot):
    """
        Make a single plot for now
    """
    # BOTH: 2 | CONTEXT: 1 | SYSTEM: 0
    executions = ["System Prompt",
                  "In-Context Examples",
                  "System Prompt & In-Context Examples"]
    try:
        #title = f"{key_info['operations'][first_plot[0]].rstrip()}() using {executions[key_info['executions'][first_plot[0]]]}"
        title = f"{key_info['operations'][first_plot[0]].rstrip()}()"
        delregex = re.compile(r"[\(\)&]")
        underregex = re.compile(r"[-\s]")
        save_name = 'v2_'+underregex.sub('_',delregex.sub('',title))+'.png'
    except:
        print("Cannot plot this data:", first_plot)
        return
    fig,ax = plt.subplots()
    n_digits = key_info['digits'][first_plot]
    negative = key_info['negatives'][first_plot]
    xlabels = [f"{'±' if neg else ''}{di}" for (di,neg) in zip(n_digits, negative)]
    x_axis = sorted(set(xlabels), key=lambda x: int(x[1:])+0.5 if '±' in x else int(x))
    x_lines = np.array([x_axis.index(xl) for xl in xlabels])
    feature  = key_info['features'][first_plot]
    # BOTH: 2 | CONTEXT: 1 | SYSTEM: 0
    lower_f   = np.intersect1d(first_plot[feature], np.where(key_info['executions'] == 0)[0])
    middle_f  = np.intersect1d(first_plot[feature], np.where(key_info['executions'] == 1)[0])
    higher_f  = np.intersect1d(first_plot[feature], np.where(key_info['executions'] == 2)[0])
    lower_nf  = np.intersect1d(first_plot[~feature], np.where(key_info['executions'] == 0)[0])
    middle_nf = np.intersect1d(first_plot[~feature], np.where(key_info['executions'] == 1)[0])
    higher_nf = np.intersect1d(first_plot[~feature], np.where(key_info['executions'] == 2)[0])
    alltogether = np.sort(np.hstack((lower_f,middle_f,higher_f,lower_nf,middle_nf,higher_nf)))
    print(alltogether)
    assert np.array_equal(alltogether,
                          first_plot)
    feature_center_line = values.iloc[middle_f]
    feature_upper, feature_lower = values.iloc[higher_f], values.iloc[lower_f]
    nofeature_center_line = values.iloc[middle_nf]
    nofeature_upper, nofeature_lower = values.iloc[higher_nf], values.iloc[lower_nf]
    #feature_linedata   = values.iloc[first_plot[feature]]
    #nofeature_linedata = values.iloc[first_plot[~feature]]
    line_names = values.columns
    legend_handles = []
    name_conversion = {'follows_regex': "Adheres to requested output format",
                       'parseable': "Answer is coherent",
                       'correct': "Answer is correctly if coherent",
                       'memorized': "Answered using value memorized from ICL",
                      }
    for idx, name in enumerate(line_names):
        """
        feature_line = ax.plot(range(len(x_axis)), feature_linedata[name], marker='.')
        feature_line = ax.plot(range(len(feature_center_line)), feature_center_line[name],
                                marker='.', markersize=12)
        ax.plot(range(len(feature_center_line)), feature_lower[name],
                                marker='v', markersize=12, color=feature_line[0].get_color())
        feature_fill = ax.fill_between(range(len(feature_center_line)), feature_lower[name], feature_upper[name],
                                      color=feature_line[0].get_color(), alpha=0.2)
        """
        #feature_line = ax.plot(range(len(feature_center_line)), feature_upper[name]+1.25,
        #                        marker='*', markersize=12)#, color=feature_line[0].get_color())
        xs = np.asarray(range(len(feature_upper[name])))*len(feature_upper[name])+idx
        feature_bars = ax.bar(xs, feature_upper[name], width=1, zorder=1)
        nofeature_bars = ax.bar(xs, nofeature_upper[name], width=0.25, zorder=2, color='k')#feature_bars[0].get_facecolor())
        """
        label=f"{name} WITHOUT features",
        nofeature_line = ax.plot(range(len(x_axis)), nofeature_linedata[name], marker='+', linestyle='--', color=feature_line[0].get_color())
        nofeature_line = ax.plot(range(len(nofeature_center_line)), nofeature_center_line[name]+2.5,
                                marker='.', markersize=12, linestyle='--', color=feature_line[0].get_color())
        nofeature_fill = ax.fill_between(range(len(nofeature_center_line)), nofeature_lower[name], nofeature_upper[name],
                                        color=nofeature_line[0].get_color(), alpha=0.2, hatch='+')
        """
        #nofeature_line = ax.plot(range(len(nofeature_center_line)), nofeature_upper[name]+3.75,
        #                        marker='*', markersize=12, linestyle='--')#, color=nofeature_line[0].get_color())
        #nofeature_vlines = []
        #for x, fun,nfun in zip(xs, feature_upper[name].to_numpy(), nofeature_upper[name].to_numpy()):
        #    nofeature_vlines.append(ax.plot([x,x],[fun,nfun], color=feature_bars[0].get_facecolor(), linewidth=2, zorder=2, edgewidth=1))
        # Add to legend
        #legend_handles.append(matplotlib.lines.Line2D([0],[0], label=name if name not in name_conversion else name_conversion[name], color=feature_line[0].get_color()))
        legend_handles.append(matplotlib.lines.Line2D([0],[0], label=name if name not in name_conversion else name_conversion[name], color=feature_bars[0].get_facecolor()))
    ax.set_xticks(len(feature_upper) * np.arange(len(x_axis)) + 3/2)
    ax.set_xticklabels(x_axis)
    ax.set_xlabel('Maximum number of digits in problem (± if integers can be negative)')
    ax.set_ylim((-0.05,1.05))
    ax.set_ylabel('Proportion')
    """
    ax.set_ylim((-0.25,5))
    bounds = [0.0,1.0, 1.25,2.25, 2.5,3.5, 3.75,4.75]
    ax.set_yticks(bounds)
    for bound in bounds:
        ax.plot(range(len(feature_center_line)), [bounds]*len(feature_center_line), color='k', linewidth=0.25, zorder=-1)
    ax.set_yticklabels([0,1]*4)
    """
    ax.set_title(title)
    # Add dummy lines to legend
    #dummy_handles = []
    #dummy_handles.append(matplotlib.lines.Line2D([0],[0], label='Prompted with Natural Language', color='k'))
    #dummy_handles.append(matplotlib.lines.Line2D([0],[0], label='Prompted with Mathematical Expression', color='k', linestyle='--'))
    #dummy_handles.append(matplotlib.lines.Line2D([0],[0], label='Prompted with ICL', color='k', marker='.'))
    #dummy_handles.append(matplotlib.lines.Line2D([0],[0], label='Prompted with ICL and System Message', color='k', marker='*'))
    #dummy_handles.append(matplotlib.patches.Patch(label="Just System Prompt (lower) to System Prompt + ICL (higher)", color='k',alpha=0.2))
    #dummy_legend = plt.legend(handles=dummy_handles, loc=4, fontsize=8, framealpha=1.0)
    legend_handles.append(matplotlib.lines.Line2D([0],[0], label='ICL uses Expression instead of Natural Language', color='k'))
    color_legend = plt.legend(handles=legend_handles, loc=1, fontsize=8, framealpha=1.0)
    #ax.add_artist(dummy_legend)
    ax.add_artist(color_legend)
    plt.tight_layout()
    plt.show()
    #fig.savefig(save_name,dpi=300)
    #print(f"Saved: {save_name}")

def time_me(func,*args,**kwargs):
    """
        Vanity/optimization/bottleneck checker
    """
    start = time.time()
    ret = func(*args,**kwargs)
    end = time.time()
    print(f"[{end-start:.6f} seconds] {func.__name__}")
    return ret

#datafiles = ['special_operation.json', 'through_multiplication.json']
datafiles = ['through_multiplication.json']
for datafile in datafiles:
    keys, values = time_me(load_json_data_as_kvs, datafile)
    print(values.iloc[0])
    key_info = time_me(parse_keys, keys)
    operations = [_ for _ in sorted(set(key_info['operations'])) if _ in ['add','mod']]
    print("Operations:",operations)
    executions = sorted(set(key_info['executions']))
    print("Executions:",executions)
    aggregate_indices = time_me(aggregate,key_info,operations,executions)
    for aggregation in aggregate_indices:
        try:
            time_me(plot_1,key_info,values,aggregation)
        except:
            print("Died")
            raise
