function tdg_write_manifest(cfg, man, versions)
% TDG_WRITE_MANIFEST  MANIFEST.csv in each version folder, from the files on disk.
%   One row per saved record: its settings, seeds, applied multisine level and the
%   file of its twin in the other version (noisy <-> noise-free). Noise-only twins
%   (twin = 1) have no noise-free pair; their "twin_of" column names the noisy record
%   they share r and f with. Rewritten completely on every call (it reflects the disk).
%   tdg_write_manifest(cfg, man, versions) writes only the named versions (D-238: a noisy-only
%   run must not rewrite the shared noise-free folder's MANIFEST.csv).
    if nargin < 3, versions = {'noisy', 'noise-free'}; end
    vers = {'noisy', cfg.out_noisy, cfg.name_free; ...
            'noise-free', cfg.out_free, cfg.name_noisy};
    for v = 1:2
        if ~any(strcmp(versions, vers{v, 1})), continue; end
        root = vers{v, 2};
        if ~exist(root, 'dir'), continue; end
        f = fullfile(root, 'MANIFEST.csv');
        fid = fopen(f, 'w');
        fprintf(fid, ['file,version,folder,record,set,block,realisation,twin,twin_of,class,Y_op,controller,' ...
                      'excitation,A_sym_N,A_anti_Nm,A_Y_N,scale,seed_ms,seed_sp,noise_seed,pair_file,MB,description\n']);
        n = 0;
        for k = 1:numel(man)
            e = man(k);
            sd = cfg.split_dir.(e.set);
            p = fullfile(root, sd, [e.file '.mat']);
            if ~isfile(p), continue; end
            S = load(p, 'meta');  m = S.meta;
            twin_of = '';
            if e.twin > 0, twin_of = sprintf('%s_r%d', e.rec, e.real); end
            pair = '';
            if e.free, pair = fullfile(vers{v, 3}, sd, [e.file '.mat']); end
            fi = dir(p);
            fprintf(fid, '%s,%s,%s,%s,%s,%s,%d,%d,%s,%s,%.4f,%s,%s,%.4f,%.4f,%.4f,%.2f,%.0f,%.0f,%.0f,%s,%.1f,"%s"\n', ...
                    e.file, vers{v, 1}, sd, e.rec, e.set, e.block, e.real, e.twin, twin_of, e.class, e.Y_op, ...
                    m.controller.name, e.p.excitation, m.amp_applied(1), m.amp_applied(2), m.amp_applied(3), ...
                    m.scale_realised, e.seed_ms, e.seed_sp, m.noise_seed, strrep(pair, '\', '/'), fi.bytes/1e6, e.desc);
            n = n + 1;
        end
        fclose(fid);
        fprintf('MANIFEST %s: %d records -> %s\n', vers{v, 1}, n, f);
    end
end
