function seed = tdg_seed(key)
% TDG_SEED  Deterministic 32-bit seed from a text key (DATA-DESIGN section 10 item 8).
%   seed = TDG_SEED('TAF|train|TR-P3|r2|nt0|ms') returns the FNV-1a hash of the key
%   (a double in [0, 2^32 - 1], valid for rng and RandStream). The key is the tuple
%   (truth, set, record, realisation, twin, signal), so no two signals of the manifest
%   can share a seed by construction of the key; tdg_manifest asserts that the hashes
%   are distinct too (a 32-bit collision among ~400 keys has probability ~2e-5).
%   Replaces the copied generator's seed = 100 * track_id + k, which collides beyond
%   100 records.
%   THEORY: FNV-1a, offset basis 2166136261, prime 16777619 (Fowler, Noll, Vo; IETF
%   draft-eastlake-fnv).
    b = uint64(unicode2native(char(key), 'UTF-8'));
    h = uint64(2166136261);
    for i = 1:numel(b)
        h = bitxor(h, b(i));
        h = mod(h * uint64(16777619), uint64(4294967296));   % < 2^57: exact in uint64
    end
    seed = double(h);
end
