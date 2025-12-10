# Merge Guide: Sanjana Branch → Main Branch

## Pre-Merge Checklist

✅ All changes committed to sanjana branch  
✅ Documentation created (SANJANA_BRANCH_CHANGES.md)  
✅ File inventory created (BRANCH_INVENTORY.json)  
✅ .gitignore updated to protect important files  
✅ All directories organized  

## Merge Strategy

### Step 1: Fetch Latest Main Branch
```bash
git fetch origin main
```

### Step 2: Review Differences
```bash
# See what files differ
git diff origin/main...HEAD --name-status

# Review specific file differences
git diff origin/main...HEAD -- FedAsync/server.py
git diff origin/main...HEAD -- FedBuff/server.py
```

### Step 3: Perform Merge
```bash
# Option A: Merge (preserves history)
git merge origin/main

# Option B: Rebase (cleaner history, but rewrites commits)
git rebase origin/main
```

### Step 4: Resolve Conflicts

If conflicts occur, prioritize sanjana branch changes for:
- `FedAsync/server.py` - CSV logging fixes
- `FedBuff/server.py` - Stagnation bug fix and CSV logging
- `utils/partitioning.py` - Leftover sample distribution fix

### Step 5: Verify Protected Items

After merge, verify these directories/files still exist:
```bash
ls -la scripts/
ls -la experiments/
ls -la hyperparameter_tuning_results/
ls -la logs/archive/
ls -la SANJANA_BRANCH_CHANGES.md
ls -la BRANCH_INVENTORY.json
```

## Files That May Have Conflicts

### High Priority (Keep Sanjana Changes)
1. **FedBuff/server.py**
   - CSV overwriting fix (always create fresh CSV)
   - Stagnation bug fix (max_rounds check in _flush_buffer)
   - Improved logging

2. **FedAsync/server.py**
   - CSV overwriting fix (always create fresh CSV)
   - Improved logging

3. **utils/partitioning.py**
   - Leftover sample distribution bug fix

### Medium Priority (Review Both)
1. **FedAsync/client.py** - ResNet-18 architecture
2. **FedAsync/run.py** - ResNet-18 architecture
3. **FedBuff/client.py** - ResNet-18 architecture
4. **FedBuff/run.py** - ResNet-18 architecture
5. **utils/model.py** - ResNet-18 build function

### Low Priority (Main Branch May Have Updates)
1. **FedAsync/config.yaml** - Configuration values
2. **FedBuff/config.yml** - Configuration values

## Protected Items (Must Not Be Deleted)

These are unique to sanjana branch and should be preserved:
- `scripts/` - All hyperparameter tuning scripts
- `experiments/squeezenet_fedbuff_results/` - Historical results
- `hyperparameter_tuning_results/` - All tuning results
- `logs/archive/` - Archived test logs
- All README.md files in these directories

## Post-Merge Verification

1. Run quick tests:
   ```bash
   python3 -m FedBuff.run
   python3 -m FedAsync.run
   ```

2. Verify CSV logging works (check logs/ directory)

3. Verify all scripts are accessible:
   ```bash
   python3 scripts/resnet18_quick_tuning.py --help
   ```

4. Check that ResNet-18 is being used:
   ```bash
   grep -r "build_resnet18" FedAsync/ FedBuff/
   ```

## Rollback Plan

If merge causes issues:
```bash
# Abort merge
git merge --abort

# Or reset to before merge
git reset --hard HEAD~1
```

## Notes

- All sanjana branch commits are dated November 25-26, 2025
- After merge, new commits will use current timeline (December 10, 2025)
- Main branch may have TrustWeight integration and other features
- Ensure ResNet-18 architecture is maintained (not reverted to SqueezeNet)

