function seed = tdg_seed(key)
% TDG_SEED  Deterministic 32-bit seed from a text key.
%   seed = TDG_SEED('TAF|train|TR-P3|r1|nt0|ms') returns the FNV-1a hash of the key (a double in
%   [0, 2^32 - 1], valid for rng and RandStream). Distinct records and signals have distinct keys;
%   tdg_manifest asserts that their seeds are distinct too.
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
