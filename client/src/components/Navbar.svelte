<script lang="ts">
  const links = [
    { href: "/", label: "Home" },
    { href: "/about", label: "About" },
    { href: "/support", label: "Support" },
  ];
  const logo = "/rna.svg";
</script>

<nav>
  <div class="nav-content">
    <div class="logo">
      <img src={logo} alt="Logo" class="logo-img h-8 w-8 mr-2" />
    </div>

    <div class="nav-links">
      {#each links as { href, label }}
        <a {href} class="nav-link">{label}</a>
      {/each}
    </div>
  </div>
</nav>

<style>
  nav {
    background-color: #1f497d;
    padding: 1rem;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
  }

  .nav-content {
    max-width: 1200px;
    margin: 0 auto;
    display: flex;
    justify-content: space-between;
    align-items: center;
  }

  .logo {
    display: flex;
    align-items: center;
  }

  .logo-img {
    height: 50px;
    width: 50px;
  }

  .nav-links {
    display: flex;
    gap: 2rem;
  }

  .nav-link {
    color: white;
    text-decoration: none;
    font-weight: 500;
    transition: color 0.2s;
  }

  .nav-link:hover {
    color: #4bacc6;
  }
</style>
