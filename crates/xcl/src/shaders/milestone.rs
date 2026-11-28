use super::{is_compute_container, Input, Pipelines};
use crate::gpu::recomp;
use crate::hle::d3d12x::real::shaders_key;
use std::collections::{HashMap, HashSet};
use std::io::{BufRead, BufReader};
use std::path::Path;
use std::sync::atomic::Ordering;
use std::time::{Duration, Instant};

const HEADER: &str = "# xcl pso manifest v1: ";
const MAX_PIPELINES: usize = 500;

pub(super) struct LaunchOptions {
    pub cutoff: u64,
    pub budget: Duration,
}

#[derive(Debug, Eq, PartialEq)]
pub(super) struct HistorySelection {
    pub compute: Vec<u64>,
    pub graphics: Vec<u64>,
}

pub(super) struct HistoryTiming {
    began: Instant,
    budget: Duration,
    count: usize,
    slow: usize,
    total: Duration,
    max: Duration,
}

impl HistoryTiming {
    pub fn new(began: Instant, budget: Duration) -> Self {
        Self { began, budget, count: 0, slow: 0, total: Duration::ZERO, max: Duration::ZERO }
    }

    pub fn record(&mut self, kind: &str, key: u64, took: Duration) {
        self.count += 1;
        self.total += took;
        self.max = self.max.max(took);
        if took >= Duration::from_secs(1) {
            self.slow += 1;
            crate::xlog_info!(
                "History preparation: {kind} {key:016x} took {:.1} s; {:.1} s elapsed, {:.1} s budget remaining",
                took.as_secs_f64(), self.began.elapsed().as_secs_f64(), self.budget.saturating_sub(self.began.elapsed()).as_secs_f64()
            );
        }
    }

    pub fn finish(&self) {
        crate::xlog_info!(
            "History preparation timing: {} pipelines, {:.1} s in calls, {:.1} s slowest, {} at least 1 s, {:.1} s wall time, {:.1} s over budget",
            self.count, self.total.as_secs_f64(), self.max.as_secs_f64(), self.slow,
            self.began.elapsed().as_secs_f64(), self.began.elapsed().saturating_sub(self.budget).as_secs_f64()
        );
    }
}

pub(super) fn launch_options(cutoff: Option<&str>, seconds: Option<&str>) -> Result<Option<LaunchOptions>, String> {
    let Some(cutoff) = cutoff else { return Ok(None) };
    let cutoff = cutoff.trim().parse::<u64>().ok().filter(|&n| n != 0)
        .ok_or("XCL_PREPARE_HISTORY_PRESENT needs a positive present count")?;
    let seconds = seconds.unwrap_or("15").trim().parse::<u64>().ok().filter(|&n| n != 0)
        .ok_or("XCL_PREPARE_HISTORY_SECONDS needs a positive number of seconds")?;
    Ok(Some(LaunchOptions { cutoff, budget: Duration::from_secs(seconds) }))
}

fn history_prefix(reader: impl BufRead, cutoff: u64) -> Result<HistorySelection, String> {
    let mut lines = reader.lines();
    let header = lines.next().transpose().map_err(|e| e.to_string())?.unwrap_or_default();
    if !header.starts_with(HEADER) {
        return Err("expected an XCL PSO manifest v1 header".into());
    }
    let (mut compute, mut graphics) = (Vec::new(), Vec::new());
    let (mut seen_compute, mut seen_graphics) = (HashSet::new(), HashSet::new());
    for (seq, row) in lines.enumerate() {
        let row = row.map_err(|e| e.to_string())?;
        let fields: Vec<_> = row.split('\t').collect();
        let invalid = || format!("invalid manifest event on line {}", seq + 2);
        if fields.first().and_then(|v| v.parse::<usize>().ok()) != Some(seq)
            || fields.get(1).and_then(|v| v.parse::<u128>().ok()).is_none() {
            return Err(invalid());
        }
        match fields[2..] {
            ["use", key, kind @ ("compute" | "graphics"), "ready" | "cold"] => {
                if key.len() != 16 || !key.bytes().all(|b| b.is_ascii_hexdigit()) {
                    return Err(invalid());
                }
                let key = u64::from_str_radix(key, 16).map_err(|_| invalid())?;
                if kind == "compute" && seen_compute.insert(key) {
                    if compute.len() + graphics.len() == MAX_PIPELINES {
                        return Err(format!("manifest prefix exceeds the {MAX_PIPELINES}-pipeline startup limit"));
                    }
                    compute.push(key);
                }
            }
            ["use", key, kind @ ("compute" | "graphics"), "ready" | "cold", recipe] => {
                if key.len() != 16 || !key.bytes().all(|b| b.is_ascii_hexdigit()) {
                    return Err(invalid());
                }
                let key = u64::from_str_radix(key, 16).map_err(|_| invalid())?;
                let recipe = match recipe {
                    "-" => None,
                    value if value.len() == 16 && value.bytes().all(|b| b.is_ascii_hexdigit()) =>
                        Some(u64::from_str_radix(value, 16).map_err(|_| invalid())?),
                    _ => return Err(invalid()),
                };
                if kind == "compute" && seen_compute.insert(key) {
                    if compute.len() + graphics.len() == MAX_PIPELINES {
                        return Err(format!("manifest prefix exceeds the {MAX_PIPELINES}-pipeline startup limit"));
                    }
                    compute.push(key);
                } else if kind == "graphics" && recipe.is_some_and(|recipe| seen_graphics.insert(recipe)) {
                    if compute.len() + graphics.len() == MAX_PIPELINES {
                        return Err(format!("manifest prefix exceeds the {MAX_PIPELINES}-pipeline startup limit"));
                    }
                    graphics.push(recipe.unwrap());
                }
            }
            ["build", _, _, _, _, _, _] => {}
            ["present", n] => {
                let n = n.parse::<u64>().map_err(|_| invalid())?;
                if n == cutoff {
                    return Ok(HistorySelection { compute, graphics });
                }
                if n > cutoff {
                    break;
                }
            }
            _ => return Err(invalid()),
        }
    }
    Err(format!("manifest has no recorded present {cutoff}"))
}

#[cfg(test)]
fn compute_prefix(reader: impl BufRead, cutoff: u64) -> Result<Vec<u64>, String> {
    Ok(history_prefix(reader, cutoff)?.compute)
}

pub(super) fn history_selection(manifest: &Path, cutoff: u64) -> Result<HistorySelection, String> {
    let file = std::fs::File::open(manifest).map_err(|e| format!("{}: {e}", manifest.display()))?;
    history_prefix(BufReader::new(file), cutoff)
}

#[cfg(test)]
fn record_indices(input: &Input, keys: &[u64]) -> Result<Vec<usize>, String> {
    Ok(record_locations(std::slice::from_ref(input), keys)?.into_iter().map(|(_, index)| index).collect())
}

fn record_locations(inputs: &[Input], keys: &[u64]) -> Result<Vec<(usize, usize)>, String> {
    let mut wanted: HashSet<_> = keys.iter().copied().collect();
    let mut found = HashMap::new();
    for (input_index, input) in inputs.iter().enumerate() {
        for index in 0..input.len() {
            if wanted.is_empty() { break }
            let (_, Some(blob)) = input.get(index) else { continue };
            let key = shaders_key(std::slice::from_ref(&blob));
            if wanted.contains(&key) && is_compute_container(&blob) {
                found.insert(key, (input_index, index));
                wanted.remove(&key);
            }
        }
        if wanted.is_empty() { break }
    }
    if !wanted.is_empty() {
        let missing: Vec<_> = keys.iter().filter(|key| wanted.contains(*key)).map(|key| format!("{key:016x}")).collect();
        return Err(format!("unmatched compute keys ({}): {}", missing.len(), missing.join(", ")));
    }
    Ok(keys.iter().map(|key| found[key]).collect())
}

fn selected_records(inputs: &[Input], keys: Vec<u64>) -> Result<Vec<(u64, usize, usize)>, String> {
    let locations = record_locations(inputs, &keys)?;
    Ok(keys.into_iter().zip(locations).map(|(key, (input, index))| (key, input, index)).collect())
}

pub(super) fn prepare_launch(
    inputs: &[Input], keys: Vec<u64>, options: &LaunchOptions, windows: u32,
    adapter: &str, progress: &super::LaunchPreparation, began: Instant, timing: &mut HistoryTiming,
) -> Result<(), String> {
    let selected = selected_records(inputs, keys)?;
    if selected.is_empty() { return Ok(()) }
    let mut pipelines = Pipelines::for_adapter(adapter)?;
    pipelines.warm_blobs = super::warm_blobs::Verified::for_device(&pipelines.dev);
    for (key, input, index) in selected.iter().copied() {
        if progress.cancel.load(Ordering::Relaxed) { return Err("Pipeline preparation cancelled".into()) }
        if began.elapsed() >= options.budget {
            crate::xlog_info!(
                "History preparation: time budget ended after {} of {} compute pipelines",
                progress.done.load(Ordering::Relaxed), selected.len()
            );
            break;
        }
        let (_, blob) = inputs[input].get(index);
        let blob = blob.ok_or_else(|| format!("cannot reread record for {key:016x}"))?;
        if shaders_key(std::slice::from_ref(&blob)) != key {
            return Err(format!("record for {key:016x} changed during preparation"));
        }
        let started = Instant::now();
        pipelines.prepare_compute(&blob, windows).map_err(|e| format!("compute {key:016x}: {e}"))?;
        timing.record("compute", key, started.elapsed());
        progress.done.fetch_add(1, Ordering::Relaxed);
    }
    if let Some(book) = &pipelines.warm_blobs { book.save(); }
    crate::xlog_info!(
        "History preparation compute outcomes: {} previously verified, {} pipeline-library/driver-blob reuses, {} uncached native creations",
        pipelines.verified.load(Ordering::Relaxed), pipelines.reused.load(Ordering::Relaxed),
        pipelines.created.load(Ordering::Relaxed)
    );
    crate::xlog_info!(
        "History preparation: {} of {} compute pipelines confirmed through present {} in {:.1} s",
        progress.done.load(Ordering::Relaxed), selected.len(), options.cutoff, began.elapsed().as_secs_f64()
    );
    Ok(())
}

pub(super) fn run(input: &Input, manifest: &Path, cutoff: u64, windows: u32, split: Option<bool>) -> Result<(), String> {
    let selected = selected_records(std::slice::from_ref(input), history_selection(manifest, cutoff)?.compute)?;
    println!("present {cutoff}: {} compute pipelines in first-use order", selected.len());
    if selected.is_empty() {
        return Ok(());
    }
    let pipelines = Pipelines::new()?;
    let split = split.unwrap_or_else(|| recomp::pso::split_wave64_forced() || recomp::runner::wave64_supported().is_ok_and(|w| !w));
    recomp::dxc::set_memory_reserve(2);
    for (key, _, index) in selected {
        let (_, blob) = input.get(index);
        let blob = blob.ok_or_else(|| format!("cannot reread record for {key:016x}"))?;
        if shaders_key(std::slice::from_ref(&blob)) != key {
            return Err(format!("record for {key:016x} changed during precompile"));
        }
        let compiled = recomp::pso::console_to_dxil_in(&blob, recomp::pso::Stage::Compute, 0, split, windows)
            .map_err(|e| format!("compute {key:016x}: {e}"))?;
        pipelines.make(&compiled);
    }
    let made = pipelines.made.load(Ordering::Relaxed);
    let reused = pipelines.reused.load(Ordering::Relaxed);
    let refused = pipelines.refused.load(Ordering::Relaxed);
    let capped = pipelines.capped.load(Ordering::Relaxed);
    println!("present {cutoff}: made {made}, including {reused} from driver blobs; refused {refused}, over cap {capped}");
    if refused + capped > 0 {
        return Err(format!("{} compute pipelines still need a runtime build", refused + capped));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const EVENTS: &str = "# xcl pso manifest v1: seq\n0\t0\tuse\t0000000000000009\tcompute\tready\n1\t0\tuse\t0000000000000002\tgraphics\tcold\n2\t1\tbuild\tcompute\t100\t-\t-\t2\t0\n3\t2\tuse\t0000000000000001\tcompute\tcold\n4\t2\tuse\t0000000000000009\tcompute\tcold\n5\t2\tpresent\t1\n6\t3\tuse\t0000000000000003\tcompute\tcold\n7\t3\tpresent\t2\n";

    #[test]
    fn selects_compute_first_uses_in_order_through_exact_present() {
        assert_eq!(compute_prefix(EVENTS.as_bytes(), 1).unwrap(), [9, 1]);
        assert_eq!(compute_prefix(EVENTS.as_bytes(), 2).unwrap(), [9, 1, 3]);
        let exact = format!("{HEADER}seq\n0\t0\tuse\t0000000000000009\tcompute\tcold\t0000000000000042\n1\t0\tpresent\t1\n");
        assert_eq!(compute_prefix(exact.as_bytes(), 1).unwrap(), [9]);
        assert!(compute_prefix(EVENTS.as_bytes(), 4).unwrap_err().contains("no recorded present 4"));
    }

    #[test]
    fn selects_exact_graphics_recipes_and_ignores_legacy_ambiguous_ones() {
        let events = format!(
            "{HEADER}seq\n0\t0\tuse\t0000000000000001\tgraphics\tcold\n1\t0\tuse\t0000000000000001\tgraphics\tcold\t00000000000000a1\n2\t0\tuse\t0000000000000002\tcompute\tready\t-\n3\t0\tuse\t0000000000000003\tgraphics\tready\t00000000000000a1\n4\t0\tuse\t0000000000000003\tgraphics\tready\t00000000000000a2\n5\t0\tpresent\t1\n"
        );
        assert_eq!(
            history_prefix(events.as_bytes(), 1).unwrap(),
            HistorySelection { compute: vec![2], graphics: vec![0xa1, 0xa2] }
        );
        let malformed = format!("{HEADER}seq\n0\t0\tuse\t0000000000000001\tgraphics\tcold\tnot-a-recipe\n1\t0\tpresent\t1\n");
        assert!(history_prefix(malformed.as_bytes(), 1).is_err());
    }

    #[test]
    fn rejects_corrupt_prefix_and_accepts_empty_prefix() {
        let empty = "# xcl pso manifest v1: seq\n0\t0\tpresent\t1\n";
        assert!(compute_prefix(empty.as_bytes(), 1).unwrap().is_empty());
        for bad in ["1\t0\tpresent\t1", "0", "0\t0\tuse\tbad\tcompute\tcold", "0\t0\tunknown"] {
            let manifest = format!("{HEADER}seq\n{bad}\n1\t1\tpresent\t1\n");
            assert!(compute_prefix(manifest.as_bytes(), 1).is_err());
        }
        assert!(compute_prefix(b"unknown\n0\t0\tpresent\t1\n".as_slice(), 1).is_err());
    }

    #[test]
    fn resolves_records_in_manifest_order_when_input_order_is_reversed() {
        let root = std::env::temp_dir().join(format!("xcl-milestone-order-{}", std::process::id()));
        std::fs::create_dir_all(&root).unwrap();
        let first = super::super::tests::compute_sample();
        let mut second = first.clone();
        *second.last_mut().unwrap() ^= 1;
        let keys = [shaders_key(std::slice::from_ref(&first)), shaders_key(std::slice::from_ref(&second))];
        let paths = [root.join("second.bin"), root.join("first.bin")];
        std::fs::write(&paths[0], second).unwrap();
        std::fs::write(&paths[1], first).unwrap();
        let result = record_indices(&Input::Files(paths.to_vec()), &keys);
        for path in paths { std::fs::remove_file(path).unwrap(); }
        std::fs::remove_dir(root).unwrap();
        assert_eq!(result.unwrap(), [1, 0]);
    }

    #[test]
    fn resolves_records_across_all_title_inputs() {
        let root = std::env::temp_dir().join(format!("xcl-milestone-inputs-{}", std::process::id()));
        std::fs::create_dir_all(&root).unwrap();
        let first = super::super::tests::compute_sample();
        let mut second = first.clone();
        *second.last_mut().unwrap() ^= 1;
        let keys = [shaders_key(std::slice::from_ref(&second)), shaders_key(std::slice::from_ref(&first))];
        let paths = [root.join("first.bin"), root.join("second.bin")];
        std::fs::write(&paths[0], first).unwrap();
        std::fs::write(&paths[1], second).unwrap();
        let inputs = [Input::Files(vec![paths[0].clone()]), Input::Files(vec![paths[1].clone()])];
        let result = record_locations(&inputs, &keys);
        for path in paths { std::fs::remove_file(path).unwrap(); }
        std::fs::remove_dir(root).unwrap();
        assert_eq!(result.unwrap(), [(1, 0), (0, 0)]);
    }

    #[test]
    fn reports_all_unmatched_keys_in_first_use_order() {
        let error = record_indices(&Input::Files(vec![]), &[9, 1]).unwrap_err();
        assert!(error.contains("0000000000000009, 0000000000000001"));
    }

    #[test]
    fn rejects_a_prefix_over_the_startup_limit() {
        let mut events = format!("{HEADER}seq\n");
        for key in 1..=MAX_PIPELINES + 1 {
            events.push_str(&format!("{}\t0\tuse\t{key:016x}\tcompute\tcold\n", key - 1));
        }
        events.push_str(&format!("{}\t0\tpresent\t1\n", MAX_PIPELINES + 1));
        assert!(compute_prefix(events.as_bytes(), 1).unwrap_err().contains("500-pipeline startup limit"));
    }

    #[test]
    fn launch_options_are_opt_in_bounded_and_strict() {
        assert!(launch_options(None, None).unwrap().is_none());
        let options = launch_options(Some("64"), None).unwrap().unwrap();
        assert_eq!(options.cutoff, 64);
        assert_eq!(options.budget, Duration::from_secs(15));
        assert_eq!(launch_options(Some("8"), Some("3")).unwrap().unwrap().budget, Duration::from_secs(3));
        for (present, seconds) in [(Some("0"), None), (Some("bad"), None), (Some("1"), Some("0")), (Some("1"), Some("bad"))] {
            assert!(launch_options(present, seconds).is_err());
        }
    }

    #[test]
    fn history_timing_tracks_slow_and_total_driver_calls() {
        let mut timing = HistoryTiming::new(Instant::now(), Duration::from_secs(5));
        timing.record("compute", 1, Duration::from_millis(900));
        timing.record("mesh", 2, Duration::from_millis(1200));
        assert_eq!(timing.count, 2);
        assert_eq!(timing.slow, 1);
        assert_eq!(timing.total, Duration::from_millis(2100));
        assert_eq!(timing.max, Duration::from_millis(1200));
    }
}
