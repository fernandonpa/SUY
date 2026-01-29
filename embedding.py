# ---------------------------------------------------------
# 4. PREPARE DATA
# ---------------------------------------------------------
# Ensure inputs are numpy arrays (if they are DataFrames)
X_train_np = X_final_e.values if hasattr(X_final_e, 'values') else X_final_e
y_train_np = y_final.values if hasattr(y_final, 'values') else y_final

X_val_np = X_val_e.values if hasattr(X_val_e, 'values') else X_val_e
y_val_np = y_val.values if hasattr(y_val, 'values') else y_val

# Scale Data (Crucial for Distance calculations)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_np)
X_val_scaled = scaler.transform(X_val_np)

# Create Triplet Dataset (It will create balanced triplets automatically)
triplet_set = TripletDataset(X_train_scaled, y_train_np)
train_loader = DataLoader(triplet_set, batch_size=256, shuffle=True)

# ---------------------------------------------------------
# 5. TRAIN LOOP
# ---------------------------------------------------------
model = EmbeddingNet(input_dim=X_train_scaled.shape[1]).to(device)
criterion = TripletLoss(margin=1.0)
optimizer = optim.Adam(model.parameters(), lr=0.001)

print("Training Siamese Network (Triplet Loss)...")
model.train()
for epoch in range(20): # Train for 20 epochs
    total_loss = 0
    for anchor, positive, negative in train_loader:
        anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass for all three
        embed_anchor = model(anchor)
        embed_pos = model(positive)
        embed_neg = model(negative)
        
        # Calculate Loss
        loss = criterion(embed_anchor, embed_pos, embed_neg)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
    print(f"Epoch {epoch+1}: Loss = {total_loss / len(train_loader):.4f}")

print("Embedding Space Learned.")
