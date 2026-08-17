% SPDX-License-Identifier: MIT
% Feed the SuperPEEC-exported .vhr files to VoxHenry's OWN parser
% (pre_input_file), the same entry point dump_vhr.m uses for the shipped
% inputs. This is the real compatibility test: our reader agreeing with our
% own writer would prove nothing.
%
% Output arity matters -- pre_input_file returns
%   [sigma_e, lambdaL, freq, dx, num_ports, pnt_lft, pnt_rght, pnt_well_cond]
% and asking for fewer silently SHIFTS the meaning of every variable (an
% earlier version of this file took 5 outputs and reported lambdaL as dx).
root = pwd();
cd([root, filesep, 'VoxHenry']);
pre_define_the_path_for_folders;
names = {'setup1', 'setup2', 'setup3'};
for i = 1:numel(names)
  f = ['setups_vhr', filesep, names{i}, '.vhr'];
  try
    [sigma_e, lambdaL, freq, dx, num_ports, pnt_lft, pnt_rght, ...
     pnt_well_cond] = pre_input_file(f);
    printf('VH-OK %-8s dims=%dx%dx%d dx=%.10g nnzsig=%d nfreq=%d nports=%d\n', ...
           names{i}, size(sigma_e,1), size(sigma_e,2), size(sigma_e,3), ...
           dx, nnz(sigma_e), numel(freq), num_ports);
    for k = 1:num_ports
      printf('        port%d: %d P-face(s), %d N-face(s)\n', ...
             k, size(pnt_lft{k},1), size(pnt_rght{k},1));
    end
  catch err
    printf('VH-FAIL %-8s %s\n', names{i}, err.message);
  end
end
cd(root);
