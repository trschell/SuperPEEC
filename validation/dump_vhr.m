% SPDX-License-Identifier: MIT
% Dump VoxHenry's own parse of every shipped .vhr input, for use as
% ground truth by validate_vhr.py (PART A).
%
% Fields chosen so that a value landing in the WRONG voxel fails as well
% as a miscount: alongside the occupied-voxel counts it records the sums
% of the conductivity and London-depth tensors and of the frequency list,
% and per-port terminal counts plus the sum of all port indices in the
% file's own 1-BASED convention (so an off-by-one in index conversion
% shows up too).
%
% Run from validation/ (checkout at validation/VoxHenry), under the Octave flatpak:
%
%   flatpak run --branch=stable --arch=x86_64 --command=/app/bin/octave \
%     --file-forwarding org.octave.Octave --no-gui --quiet dump_vhr.m
%
% Writes vhr_ref.txt beside this script. Nothing is written inside the
% VoxHenry checkout, so its git tree stays clean.

root = pwd;
outfile = [root, filesep, 'vhr_ref.txt'];
cd([root, filesep, 'VoxHenry']);
pre_define_the_path_for_folders;

l = dir(['Input_files', filesep, '*.vhr']);
fid = fopen(outfile, 'w');
for i = 1:numel(l)
  f = ['Input_files', filesep, l(i).name];
  [sigma_e, lambdaL, freq, dx, num_ports, pnt_lft, pnt_rght, ...
   pnt_well_cond] = pre_input_file(f);
  fprintf(fid, 'FILE %s\n', l(i).name);
  fprintf(fid, 'DIMS %d %d %d\n', size(sigma_e,1), size(sigma_e,2), ...
          size(sigma_e,3));
  fprintf(fid, 'DX %.17g\n', dx);
  fprintf(fid, 'NNZSIG %d\n', nnz(sigma_e));
  fprintf(fid, 'SUMSIG %.17g\n', sum(sigma_e(:)));
  if isempty(lambdaL)
    fprintf(fid, 'NNZLAM -1\nSUMLAM 0\n');
  else
    fprintf(fid, 'NNZLAM %d\nSUMLAM %.17g\n', nnz(lambdaL), sum(lambdaL(:)));
  end
  fprintf(fid, 'NFREQ %d\n', numel(freq));
  fprintf(fid, 'FREQSUM %.17g\n', sum(freq(:)));
  fprintf(fid, 'NPORTS %d\n', num_ports);
  for p = 1:num_ports
    a = pnt_lft{p};
    b = pnt_rght{p};
    fprintf(fid, 'PORT %d NP %d NN %d SUMP %d SUMN %d\n', p, ...
            size(a,1), size(b,1), sum(a(:)), sum(b(:)));
  end
  fprintf(fid, 'NGRND %d\n', size(pnt_well_cond,1));
end
fclose(fid);
cd(root);
disp(['Wrote ', outfile]);
