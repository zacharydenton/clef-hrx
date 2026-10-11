//! Compact in-process keys; the compiler still owns its artifact cache.
use hrx::loom::Specialization;
use sha2::{Digest, Sha256};
use std::collections::HashMap;

pub(crate) type KernelKey = ([u8; 32], String);

#[derive(Default)]
pub(crate) struct SourceFingerprints {
    // Addresses accelerate lookup, but never establish identity by themselves:
    // benchmark callers may reuse an allocation for different source text.
    entries: HashMap<(usize, usize), (Box<str>, [u8; 32])>,
}

impl SourceFingerprints {
    fn fingerprint(&mut self, source: &str) -> [u8; 32] {
        let address = (source.as_ptr() as usize, source.len());
        if let Some((saved, digest)) = self.entries.get(&address)
            && saved.as_ref() == source
        {
            return *digest;
        }
        let digest = Sha256::digest(source.as_bytes()).into();
        if self.entries.len() >= 64 {
            self.entries.clear();
        }
        self.entries.insert(address, (source.into(), digest));
        digest
    }

    pub(crate) fn key(&mut self, source: &str, spec: &Specialization) -> KernelKey {
        (
            self.fingerprint(source),
            format!("{:?}{:?}", spec.symbol(), spec.configuration()),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reusing_a_source_allocation_does_not_reuse_its_identity() {
        let mut cache = SourceFingerprints::default();
        let spec = Specialization::new("entry");
        let mut source = String::from("first source");
        let address = source.as_ptr();
        let first = cache.key(&source, &spec);
        source.replace_range(.., "other source");
        assert_eq!(source.as_ptr(), address);
        assert_ne!(cache.key(&source, &spec), first);
        // Identical content has the same key even in a different allocation.
        assert_eq!(cache.key("first source", &spec), first);
    }

    #[test]
    fn exports_and_configuration_are_part_of_the_key() {
        let mut cache = SourceFingerprints::default();
        let mut spec = Specialization::new("first");
        let first = cache.key("module", &spec);
        spec.set_symbol("second");
        assert_ne!(cache.key("module", &spec), first);
        spec.set_symbol("first");
        spec.set_config("width", "128");
        assert_ne!(cache.key("module", &spec), first);
    }
}
